from utils.config import load_config
from train_utils.peft_utils import add_lora
from models.planner import PlannerModel, load_model_and_tokenizer
from robot_fm_data.data import PlannerDataset

from transformers import (
    Trainer,
    TrainingArguments,
)
from robot_fm_data.format import AlpacaFormatter, ChatMLFormatter
from robot_fm_data.loader import load_dataset, create_train_eval_split
from transformers import set_seed
from pathlib import Path
# --------------------------------------------------
# Load config
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent
cfg = load_config(
    str(ROOT / "config" / "train.yaml")
)

#set seed for reproducibility
set_seed(cfg["seed"])

# --------------------------------------------------
# Load model
# --------------------------------------------------

model, tokenizer = load_model_and_tokenizer(model_name=cfg["model"]["name"])

"""PlannerModel(
    model_name=cfg["model"]["name"]
)"""

tokenizer.pad_token = tokenizer.eos_token
model.llm.config.pad_token_id = tokenizer.pad_token_id

# --------------------------------------------------
# LoRA
# --------------------------------------------------

if cfg["use_lora"]:
    model.llm = add_lora(
        model.llm,
        cfg["lora"]
    )
    model.llm.print_trainable_parameters()

# --------------------------------------------------
# Data format
# --------------------------------------------------

if cfg["dataset"]["format"] == "alpaca":
    formatter = AlpacaFormatter()

elif cfg["dataset"]["format"] == "chatml":
    formatter = ChatMLFormatter(tokenizer)
    #raise ValueError("Unknown dataset format")


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

dataset_cfg = cfg["datasets"][0]

dataset_path = dataset_cfg["path"]
dataset_type = dataset_cfg["type"]

data = load_dataset(dataset_cfg)

print(f"Loaded {len(data)} samples")

print("Example:")
print(data[0])

print("Formatted example:")
prompt, response = formatter.format(data[0])
print(prompt)
print(response)

train_data, eval_data = create_train_eval_split(
    data,
    eval_ratio=cfg["training"]["eval_ratio"],
    seed=cfg["seed"],
)

print(f"Length of train data: {len(train_data)}")
print(f"Length of eval data: {len(eval_data)}")

train_dataset = PlannerDataset(
    data=train_data,
    tokenizer=tokenizer,
    formatter=formatter,
    max_length=cfg["training"]["max_length"],
)

eval_dataset = PlannerDataset(
    data=eval_data,
    tokenizer=tokenizer,
    formatter=formatter,
    max_length=cfg["training"]["max_length"],
)

# --------------------------------------------------
# Training arguments
# --------------------------------------------------

from datetime import datetime


model_name = cfg["model"]["name"].split("/")[-1]


run_name = (
    f"{model_name}"
    f"_lr{cfg['training']['lr']}"
    f"_r{cfg['lora']['r']}"
    f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)


#for checkpoints
checkpoint_dir = (f"{cfg['output']['checkpoint_root']}/{run_name}")


#for final model - chosen based on best eval results.
best_adapter_dir = (
    f"{checkpoint_dir}/best_adapter"
)

steps_per_epoch = (
    len(train_dataset)
    // (
        cfg["training"]["batch_size"]
        * cfg["training"]["grad_accum"]
    )
)

eval_steps = max(1, steps_per_epoch // 4)
training_args = TrainingArguments(

    output_dir = checkpoint_dir,

    per_device_train_batch_size=cfg["training"]["batch_size"],

    gradient_accumulation_steps=cfg["training"]["grad_accum"],

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

)

# --------------------------------------------------
# Log
# --------------------------------------------------
total_params = sum(
    p.numel() for p in model.llm.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.llm.parameters()
    if p.requires_grad
)

print(f"Trainable Parameters: {trainable_params:,}")
print(f"Total Parameters: {total_params:,}")
print(
    f"Trainable %: "
    f"{100 * trainable_params / total_params:.4f}%"
)

#create directories for loggign checkpoints
import os

os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(best_adapter_dir, exist_ok=True)

# --------------------------------------------------
# Trainer
# --------------------------------------------------

trainer = Trainer(
    model=model.llm,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# --------------------------------------------------
# Train
# --------------------------------------------------

trainer.train()

# --------------------------------------------------
# Save model
# --------------------------------------------------

trainer.save_model(best_adapter_dir)

tokenizer.save_pretrained(best_adapter_dir)