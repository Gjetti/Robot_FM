from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import (
    Trainer,
    TrainingArguments,
    set_seed,
)

from utils.config import load_config
from train_utils.peft_utils import add_lora
from models.planner import load_model_and_tokenizer
from robot_fm_data.data import PlannerDataset
from robot_fm_data.format import AlpacaFormatter, ChatMLFormatter
from robot_fm_data.loader import load_dataset, create_train_eval_split


LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_example_value(
    example: Any,
    key: str,
) -> Any:
    """
    Read a value from an example.

    The adjusted ChatML JSONL stores planner data at:
        example["messages"]

    and RA-BC information at:
        example["metadata"]["segment_return_to_go"]

    For backward compatibility, this helper also accepts values stored
    directly at the top level.
    """
    if isinstance(example, dict):
        # Backward compatibility with older datasets.
        if key in example:
            return example[key]

        metadata = example.get("metadata")

        if isinstance(metadata, dict) and key in metadata:
            return metadata[key]

        raise KeyError(
            f"Example does not contain {key!r} at the top level "
            "or inside example['metadata']. "
            f"Top-level keys: {list(example.keys())}; "
            f"metadata keys: "
            f"{list(metadata.keys()) if isinstance(metadata, dict) else None}"
        )

    if hasattr(example, key):
        return getattr(example, key)

    metadata = getattr(example, "metadata", None)

    if isinstance(metadata, dict) and key in metadata:
        return metadata[key]

    try:
        return example[key]
    except Exception as exc:
        raise KeyError(
            f"Could not read {key!r} from the example or its metadata."
        ) from exc


def fit_return_range(
    data,
    return_column: str,
    lower_percentile: float = 0.0,
    upper_percentile: float = 100.0,
) -> tuple[float, float]:
    """
    Fit the return normalization range using only the training split.

    Percentile clipping can later be used if a few return outliers
    stretch the range too strongly.
    """
    returns = np.asarray(
        [
            float(get_example_value(example, return_column))
            for example in data
        ],
        dtype=np.float64,
    )

    if len(returns) == 0:
        raise ValueError("Cannot fit return weights on an empty dataset.")

    if not np.all(np.isfinite(returns)):
        invalid_count = int((~np.isfinite(returns)).sum())
        raise ValueError(
            f"Found {invalid_count} non-finite values in "
            f"{return_column!r}."
        )

    return_min = float(
        np.percentile(returns, lower_percentile)
    )
    return_max = float(
        np.percentile(returns, upper_percentile)
    )

    return return_min, return_max


def return_to_weight(
    return_to_go: float,
    return_min: float,
    return_max: float,
    min_weight: float,
    max_weight: float,
) -> float:
    """
    Map return-to-go to a bounded, non-negative BC weight.

    return_min -> min_weight
    return_max -> max_weight
    """
    if min_weight < 0:
        raise ValueError("min_weight must be non-negative.")

    if max_weight <= 0:
        raise ValueError("max_weight must be positive.")

    if max_weight < min_weight:
        raise ValueError(
            "max_weight must be greater than or equal to min_weight."
        )

    value = float(return_to_go)

    if return_max <= return_min:
        # All training returns are effectively identical.
        return float(max_weight)

    value = float(
        np.clip(
            value,
            return_min,
            return_max,
        )
    )

    normalized_return = (
        (value - return_min)
        / (return_max - return_min)
    )

    weight = (
        min_weight
        + normalized_return
        * (max_weight - min_weight)
    )

    return float(weight)


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class RewardAlignedPlannerDataset(Dataset):
    """
    Wrap PlannerDataset and add one scalar sample weight per row.

    Each raw row already contains the segment return-to-go. Therefore,
    every state inside the same segment naturally receives the same
    return-derived weight.
    """

    def __init__(
        self,
        data,
        tokenizer,
        formatter,
        max_length: int,
        return_column: str,
        return_min: float,
        return_max: float,
        min_weight: float,
        max_weight: float,
    ):
        self.raw_data = data

        self.base_dataset = PlannerDataset(
            data=data,
            tokenizer=tokenizer,
            formatter=formatter,
            max_length=max_length,
        )

        self.return_column = return_column
        self.return_min = float(return_min)
        self.return_max = float(return_max)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(
        self,
        index: int,
    ) -> Dict[str, torch.Tensor]:
        item = dict(self.base_dataset[index])

        return_to_go = float(
            get_example_value(
                self.raw_data[index],
                self.return_column,
            )
        )

        sample_weight = return_to_weight(
            return_to_go=return_to_go,
            return_min=self.return_min,
            return_max=self.return_max,
            min_weight=self.min_weight,
            max_weight=self.max_weight,
        )

        item["sample_weight"] = torch.tensor(
            sample_weight,
            dtype=torch.float32,
        )

        return item


# ---------------------------------------------------------------------------
# Return-aligned Trainer
# ---------------------------------------------------------------------------

class RewardAlignedTrainer(Trainer):
    """
    Trainer using return-aligned causal-language-model cross-entropy.

    The loss is:
      1. averaged across valid response tokens for each example;
      2. weighted by the example's segment return-to-go weight;
      3. normalized by the sum of weights in the batch.

      Check SARM
    """

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        sample_weights = inputs.pop("sample_weight")
        labels = inputs.pop("labels")

        outputs = model(
            **inputs,
            use_cache=False,
        )

        logits = outputs.logits

        # Causal LM shifting:
        # logits[:, t] predicts labels[:, t + 1].
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        batch_size = shift_logits.shape[0]
        sequence_length = shift_logits.shape[1]
        vocabulary_size = shift_logits.shape[2]

        token_losses = F.cross_entropy(
            shift_logits.view(
                batch_size * sequence_length,
                vocabulary_size,
            ),
            shift_labels.view(
                batch_size * sequence_length,
            ),
            reduction="none",
            ignore_index=-100,
        ).view(
            batch_size,
            sequence_length,
        )

        valid_token_mask = (
            shift_labels != -100
        ).to(
            dtype=token_losses.dtype
        )

        valid_token_counts = (
            valid_token_mask
            .sum(dim=1)
            .clamp_min(1.0)
        )

        # One average response-token CE value per planner row.
        per_example_losses = (
            token_losses
            * valid_token_mask
        ).sum(
            dim=1
        ) / valid_token_counts

        sample_weights = sample_weights.to(
            device=per_example_losses.device,
            dtype=per_example_losses.dtype,
        ).view(-1)

        if sample_weights.shape[0] != batch_size:
            raise ValueError(
                "Number of sample weights does not match batch size: "
                f"{sample_weights.shape[0]} versus {batch_size}."
            )

        if torch.any(sample_weights < 0):
            raise ValueError(
                "Reward-aligned BC weights must be non-negative."
            )

        loss = (
            sample_weights
            * per_example_losses
        ).sum() / (
            sample_weights.sum().clamp_min(1e-8)
        )

        if return_outputs:
            return loss, outputs

        return loss


# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

cfg = load_config(
    str(ROOT / "config" / "train_RA-BC.yaml")
)

set_seed(cfg["seed"])


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

model, tokenizer = load_model_and_tokenizer(
    model_name=cfg["model"]["name"]
)

tokenizer.pad_token = tokenizer.eos_token
model.llm.config.pad_token_id = tokenizer.pad_token_id


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

if cfg["use_lora"]:
    model.llm = add_lora(
        model.llm,
        cfg["lora"],
    )

    if LOCAL_RANK == 0:
        model.llm.print_trainable_parameters()


# ---------------------------------------------------------------------------
# Data format
# ---------------------------------------------------------------------------

if cfg["dataset"]["format"] == "alpaca":
    formatter = AlpacaFormatter()

elif cfg["dataset"]["format"] == "chatml":
    formatter = ChatMLFormatter(tokenizer)

else:
    raise ValueError(
        f"Unknown dataset format: "
        f"{cfg['dataset']['format']!r}"
    )


# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------

dataset_cfg = cfg["datasets"][0]

data = load_dataset(dataset_cfg)

if len(data) == 0:
    raise ValueError("The loaded dataset is empty.")

# Validate that the existing loader preserved the JSONL metadata.
return_column_for_validation = cfg["reward_alignment"]["return_column"]

try:
    first_return = float(
        get_example_value(
            data[0],
            return_column_for_validation,
        )
    )
except Exception as exc:
    raise RuntimeError(
        "The dataset loader did not preserve the RA-BC metadata. "
        "The JSONL example must remain a dictionary containing "
        "'messages' and 'metadata'."
    ) from exc

if LOCAL_RANK == 0:
    print(f"Loaded {len(data)} samples")

    print("\nRaw example:")
    print(data[0])

    print("\nRA-BC metadata check:")
    print(
        f"{return_column_for_validation} = "
        f"{first_return}"
    )

    print("\nFormatted example:")
    prompt, response = formatter.format(data[0])
    print(prompt)
    print(response)


train_data, eval_data = create_train_eval_split(
    data,
    eval_ratio=cfg["training"]["eval_ratio"],
    seed=cfg["seed"],
)

if LOCAL_RANK == 0:
    print(f"\nLength of train data: {len(train_data)}")
    print(f"Length of eval data : {len(eval_data)}")


# ---------------------------------------------------------------------------
# Return-alignment configuration
# ---------------------------------------------------------------------------

return_column = cfg["reward_alignment"]["return_column"]

return_min, return_max = fit_return_range(
    data=train_data,
    return_column=return_column,
    lower_percentile=cfg["reward_alignment"].get(
        "lower_percentile",
        0.0,
    ),
    upper_percentile=cfg["reward_alignment"].get(
        "upper_percentile",
        100.0,
    ),
)

if LOCAL_RANK == 0:
    train_returns = np.asarray(
        [
            float(
                get_example_value(
                    example,
                    return_column,
                )
            )
            for example in train_data
        ]
    )

    print("\n==============================")
    print("REWARD ALIGNMENT")
    print("==============================")
    print(f"Return column : {return_column}")
    print(
        f"Observed train return range : "
        f"[{train_returns.min():.6f}, "
        f"{train_returns.max():.6f}]"
    )
    print(
        f"Weight normalization range  : "
        f"[{return_min:.6f}, {return_max:.6f}]"
    )
    print(
        f"Weight range                : "
        f"[{cfg['reward_alignment']['min_weight']}, "
        f"{cfg['reward_alignment']['max_weight']}]"
    )


# ---------------------------------------------------------------------------
# Build weighted datasets
# ---------------------------------------------------------------------------

train_dataset = RewardAlignedPlannerDataset(
    data=train_data,
    tokenizer=tokenizer,
    formatter=formatter,
    max_length=cfg["training"]["max_length"],
    return_column=return_column,
    return_min=return_min,
    return_max=return_max,
    min_weight=cfg["reward_alignment"]["min_weight"],
    max_weight=cfg["reward_alignment"]["max_weight"],
)

eval_dataset = RewardAlignedPlannerDataset(
    data=eval_data,
    tokenizer=tokenizer,
    formatter=formatter,
    max_length=cfg["training"]["max_length"],
    return_column=return_column,

    # Use the training range for evaluation too.
    return_min=return_min,
    return_max=return_max,

    min_weight=cfg["reward_alignment"]["min_weight"],
    max_weight=cfg["reward_alignment"]["max_weight"],
)


if LOCAL_RANK == 0:
    print("\nWeighted dataset samples:")

    for index in range(min(3, len(train_dataset))):
        raw_return = float(
            get_example_value(
                train_data[index],
                return_column,
            )
        )

        weighted_item = train_dataset[index]

        print(
            f"sample={index}, "
            f"return={raw_return:.6f}, "
            f"weight={float(weighted_item['sample_weight']):.6f}"
        )


# ---------------------------------------------------------------------------
# Run and checkpoint names
# ---------------------------------------------------------------------------

model_name = cfg["model"]["name"].split("/")[-1]

run_name = (
    f"RA-BC_"
    f"{model_name}"
    f"_lr{cfg['training']['lr']}"
    f"_r{cfg['lora']['r']}"
    f"_w{cfg['reward_alignment']['min_weight']}"
    f"-{cfg['reward_alignment']['max_weight']}"
    f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)

checkpoint_dir = (
    f"{cfg['output']['checkpoint_root']}/{run_name}"
)

best_adapter_dir = (
    f"{checkpoint_dir}/best_adapter"
)


# ---------------------------------------------------------------------------
# Training arguments
# ---------------------------------------------------------------------------

world_size = int(
    os.environ.get(
        "WORLD_SIZE",
        1,
    )
)

if LOCAL_RANK == 0:
    print("WORLD_SIZE =", world_size)

steps_per_epoch = max(
    1,
    len(train_dataset)
    // (
        cfg["training"]["batch_size"]
        * cfg["training"]["grad_accum"]
        * world_size
    ),
)

eval_steps = max(
    1,
    steps_per_epoch // 4,
)

training_args = TrainingArguments(
    output_dir=checkpoint_dir,

    per_device_train_batch_size=(
        cfg["training"]["batch_size"]
    ),

    per_device_eval_batch_size=(
        cfg["training"].get(
            "eval_batch_size",
            cfg["training"]["batch_size"],
        )
    ),

    gradient_accumulation_steps=(
        cfg["training"]["grad_accum"]
    ),

    optim=cfg["optimizer"]["name"],

    learning_rate=cfg["training"]["lr"],

    lr_scheduler_type=cfg["scheduler"]["name"],

    warmup_ratio=cfg["scheduler"]["warmup_ratio"],

    num_train_epochs=cfg["training"]["epochs"],

    bf16=cfg["training"]["bf16"],

    logging_steps=cfg["training"]["logging_steps"],

    eval_strategy="steps",
    eval_steps=eval_steps,

    save_strategy="steps",
    save_steps=eval_steps,
    save_total_limit=4,

    report_to="tensorboard",
    logging_dir=f"{checkpoint_dir}/tensorboard",

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    ddp_find_unused_parameters=False,

    # Required so Trainer does not remove sample_weight before
    # RewardAlignedTrainer.compute_loss receives it.
    remove_unused_columns=False,
)


# ---------------------------------------------------------------------------
# Parameter logging
# ---------------------------------------------------------------------------

total_params = sum(
    parameter.numel()
    for parameter in model.llm.parameters()
)

trainable_params = sum(
    parameter.numel()
    for parameter in model.llm.parameters()
    if parameter.requires_grad
)

if LOCAL_RANK == 0:
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Total Parameters    : {total_params:,}")
    print(
        "Trainable %         : "
        f"{100 * trainable_params / total_params:.4f}%"
    )


# ---------------------------------------------------------------------------
# Directories and model settings
# ---------------------------------------------------------------------------

os.makedirs(
    checkpoint_dir,
    exist_ok=True,
)

os.makedirs(
    best_adapter_dir,
    exist_ok=True,
)

model.llm.config.use_cache = False
model.llm.enable_input_require_grads()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

trainer = RewardAlignedTrainer(
    model=model.llm,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

trainer.train()


# ---------------------------------------------------------------------------
# Save best model
# ---------------------------------------------------------------------------

if trainer.is_world_process_zero():
    trainer.save_model(best_adapter_dir)
    tokenizer.save_pretrained(best_adapter_dir)