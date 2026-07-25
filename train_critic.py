from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.stats import pearsonr, spearmanr
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from models.critic_structured_task import (
    CriticConfig,
    StateActionTransformerCritic,
)
from robot_fm_data.critic_data import (
    CriticDataConfig,
    create_critic_dataloaders,
)

ROOT = Path(__file__).resolve().parent


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}.")
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is unavailable.")
    return device


def move_inputs(inputs: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {
        name: tensor.to(device=device, non_blocking=True)
        for name, tensor in inputs.items()
    }


def compute_loss(
    prediction: Tensor,
    target: Tensor,
    loss_name: str,
    huber_delta: float,
) -> Tensor:
    if loss_name == "mse":
        return F.mse_loss(prediction, target)
    if loss_name == "huber":
        return F.smooth_l1_loss(prediction, target, beta=huber_delta)
    raise ValueError(f"Unsupported loss {loss_name!r}. Use 'mse' or 'huber'.")


def make_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    minimum_lr_ratio: float,
) -> LambdaLR:
    def multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        decay_steps = max(total_steps - warmup_steps, 1)
        progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_lr_ratio + (1.0 - minimum_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=multiplier)


def safe_correlation(target: np.ndarray, prediction: np.ndarray, kind: str) -> float:
    if target.size < 2 or np.allclose(target, target[0]) or np.allclose(prediction, prediction[0]):
        return float("nan")
    if kind == "pearson":
        return float(pearsonr(target, prediction).statistic)
    if kind == "spearman":
        return float(spearmanr(target, prediction).statistic)
    raise ValueError(f"Unknown correlation kind: {kind}")


def pairwise_ranking_accuracy(
    target: np.ndarray,
    prediction: np.ndarray,
    maximum_pairs: int,
    tie_tolerance: float,
    seed: int,
) -> float:
    """Fraction of non-tied pairs whose predicted ordering is correct."""
    n = target.size
    if n < 2:
        return float("nan")

    total_pairs = n * (n - 1) // 2
    rng = np.random.default_rng(seed)

    if total_pairs <= maximum_pairs:
        left, right = np.triu_indices(n, k=1)
    else:
        left = rng.integers(0, n, size=maximum_pairs)
        right = rng.integers(0, n, size=maximum_pairs)
        keep = left != right
        left, right = left[keep], right[keep]

    target_difference = target[left] - target[right]
    prediction_difference = prediction[left] - prediction[right]
    non_tied = np.abs(target_difference) > tie_tolerance

    if not np.any(non_tied):
        return float("nan")

    correct = (
        np.sign(target_difference[non_tied])
        == np.sign(prediction_difference[non_tied])
    )
    return float(np.mean(correct))


@torch.no_grad()
def evaluate(
    model: StateActionTransformerCritic,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_name: str,
    huber_delta: float,
    ranking_max_pairs: int,
    ranking_tie_tolerance: float,
    seed: int,
    autocast_enabled: bool,
    autocast_dtype: torch.dtype,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    predictions: list[Tensor] = []
    targets: list[Tensor] = []

    for batch in data_loader:
        model_inputs = move_inputs(batch["model_inputs"], device)
        q_target = batch["q_target"].to(device=device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=autocast_enabled,
        ):
            q_prediction = model(**model_inputs).q_value
            loss = compute_loss(q_prediction, q_target, loss_name, huber_delta)

        batch_size = q_target.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
        predictions.append(q_prediction.detach().float().cpu())
        targets.append(q_target.detach().float().cpu())

    if total_examples == 0:
        raise RuntimeError("The evaluation DataLoader produced no samples.")

    prediction = torch.cat(predictions).numpy()
    target = torch.cat(targets).numpy()
    errors = prediction - target

    metrics = {
        "loss": total_loss / total_examples,
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "pearson": safe_correlation(target, prediction, "pearson"),
        "spearman": safe_correlation(target, prediction, "spearman"),
        "pairwise_ranking_accuracy": pairwise_ranking_accuracy(
            target,
            prediction,
            maximum_pairs=ranking_max_pairs,
            tie_tolerance=ranking_tie_tolerance,
            seed=seed,
        ),
        "prediction_mean": float(np.mean(prediction)),
        "prediction_std": float(np.std(prediction)),
        "target_mean": float(np.mean(target)),
        "target_std": float(np.std(target)),
    }
    return metrics, prediction, target


def build_checkpoint(
    model: StateActionTransformerCritic,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    global_step: int,
    best_eval_loss: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_eval_loss": best_eval_loss,
        "config": config,
    }


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def rotate_checkpoints(directory: Path, save_total_limit: int) -> None:
    if save_total_limit <= 0:
        return
    checkpoints = sorted(
        directory.glob("checkpoint-epoch*-step*.pt"),
        key=lambda path: path.stat().st_mtime,
    )
    while len(checkpoints) > save_total_limit:
        checkpoints.pop(0).unlink(missing_ok=True)


def load_checkpoint(
    path: Path,
    model: StateActionTransformerCritic,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> tuple[int, int, float]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return (
        int(checkpoint["epoch"]) + 1,
        int(checkpoint["global_step"]),
        float(checkpoint.get("best_eval_loss", float("inf"))),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the structured Q critic.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "train_critic.yaml",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Optional checkpoint path; overrides training.resume_from.",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_yaml(config_path)
    seed = int(config["seed"])
    set_seed(seed)
    device = select_device(str(config["training"]["device"]))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"{config['output'].get('run_name', 'critic')}"
        f"_lr{config['training']['learning_rate']}"
        f"_{timestamp}"
    )
    checkpoint_root = (ROOT / Path(config["output"]["checkpoint_root"])).resolve()
    run_directory = checkpoint_root / run_name
    checkpoint_directory = run_directory / "checkpoints"
    tensorboard_directory = run_directory / "tensorboard"
    best_model_directory = run_directory / "best_model"

    for directory in (checkpoint_directory, tensorboard_directory, best_model_directory):
        directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, run_directory / "train_critic.yaml")

    data_config = CriticDataConfig(
        row_csv=str(config["dataset"]["row_csv"]),
        segment_csv=str(config["dataset"]["segment_csv"]),
        base_csv=str(config["dataset"]["base_csv"]),
        max_objects=int(config["dataset"]["max_objects"]),
        gamma=float(config["dataset"]["gamma"]),
    )
    train_loader, eval_loader = create_critic_dataloaders(
        config=data_config,
        batch_size=int(config["training"]["batch_size"]),
        eval_ratio=float(config["dataset"]["eval_ratio"]),
        seed=seed,
        num_workers=int(config["dataset"]["num_workers"]),
    )

    if len(train_loader) == 0 or len(eval_loader) == 0:
        raise RuntimeError("A training or evaluation DataLoader contains no batches.")

    critic_config = CriticConfig(**config["model"])
    model = StateActionTransformerCritic(critic_config).to(device)

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    print("=" * 60)
    print("CRITIC TRAINING")
    print("=" * 60)
    print(f"Run directory: {run_directory}")
    print(f"Device: {device}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Evaluation batches: {len(eval_loader)}")
    print(f"Total parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")

    train_cfg = config["training"]
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        betas=tuple(float(value) for value in train_cfg["betas"]),
        eps=float(train_cfg["epsilon"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    grad_accum = int(train_cfg["gradient_accumulation_steps"])
    epochs = int(train_cfg["epochs"])
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_optimizer_steps = optimizer_steps_per_epoch * epochs
    warmup_steps = int(
        round(total_optimizer_steps * float(config["scheduler"]["warmup_ratio"]))
    )
    scheduler = make_cosine_scheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_optimizer_steps,
        minimum_lr_ratio=float(config["scheduler"]["minimum_lr_ratio"]),
    )

    precision = str(train_cfg["precision"]).lower()
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision must be fp32, fp16, or bf16.")
    autocast_enabled = device.type == "cuda" and precision != "fp32"
    autocast_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(
        enabled=device.type == "cuda" and precision == "fp16"
    )

    start_epoch = 0
    global_step = 0
    best_eval_loss = float("inf")
    configured_resume = train_cfg.get("resume_from")
    resume_path = args.resume or (Path(configured_resume) if configured_resume else None)
    if resume_path is not None:
        start_epoch, global_step, best_eval_loss = load_checkpoint(
            resume_path.resolve(), model, optimizer, scheduler, scaler, device
        )
        print(f"Resumed at epoch {start_epoch} from {resume_path}.")

    writer = SummaryWriter(log_dir=str(tensorboard_directory))
    writer.add_text(
        "run/config",
        "```yaml\n" + yaml.safe_dump(config, sort_keys=False) + "\n```",
        0,
    )

    loss_name = str(train_cfg["loss"]).lower()
    huber_delta = float(train_cfg["huber_delta"])
    max_grad_norm = float(train_cfg["max_grad_norm"])
    logging_steps = int(train_cfg["logging_steps"])
    evaluation_every_epochs = int(train_cfg["evaluation_every_epochs"])
    save_every_epochs = int(train_cfg["save_every_epochs"])
    save_total_limit = int(config["output"]["save_total_limit"])

    optimizer.zero_grad(set_to_none=True)

    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            epoch_loss_sum = 0.0
            epoch_examples = 0
            accumulation_counter = 0

            for batch_index, batch in enumerate(train_loader):
                model_inputs = move_inputs(batch["model_inputs"], device)
                q_target = batch["q_target"].to(device=device, non_blocking=True)

                with torch.autocast(
                    device_type=device.type,
                    dtype=autocast_dtype,
                    enabled=autocast_enabled,
                ):
                    q_prediction = model(**model_inputs).q_value
                    raw_loss = compute_loss(
                        q_prediction,
                        q_target,
                        loss_name,
                        huber_delta,
                    )
                    loss = raw_loss / grad_accum

                scaler.scale(loss).backward()
                batch_size = q_target.shape[0]
                epoch_loss_sum += float(raw_loss.item()) * batch_size
                epoch_examples += batch_size
                accumulation_counter += 1

                last_batch = batch_index == len(train_loader) - 1
                if accumulation_counter != grad_accum and not last_batch:
                    continue

                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                accumulation_counter = 0
                global_step += 1

                if logging_steps > 0 and global_step % logging_steps == 0:
                    writer.add_scalar("train/batch_loss", float(raw_loss.item()), global_step)
                    writer.add_scalar(
                        "train/learning_rate", optimizer.param_groups[0]["lr"], global_step
                    )
                    writer.add_scalar("train/gradient_norm", float(gradient_norm), global_step)
                    writer.add_scalar(
                        "train/q_prediction_mean",
                        float(q_prediction.detach().mean().item()),
                        global_step,
                    )
                    writer.add_scalar(
                        "train/q_target_mean",
                        float(q_target.detach().mean().item()),
                        global_step,
                    )

            train_epoch_loss = epoch_loss_sum / epoch_examples
            writer.add_scalar("train/epoch_loss", train_epoch_loss, epoch + 1)
            print(
                f"Epoch {epoch + 1:03d}/{epochs:03d} "
                f"| train_loss={train_epoch_loss:.6f}"
            )

            should_evaluate = (
                (epoch + 1) % evaluation_every_epochs == 0
                or epoch + 1 == epochs
            )
            if should_evaluate:
                metrics, predictions, targets = evaluate(
                    model=model,
                    data_loader=eval_loader,
                    device=device,
                    loss_name=loss_name,
                    huber_delta=huber_delta,
                    ranking_max_pairs=int(config["evaluation"]["ranking_max_pairs"]),
                    ranking_tie_tolerance=float(
                        config["evaluation"]["ranking_tie_tolerance"]
                    ),
                    seed=seed,
                    autocast_enabled=autocast_enabled,
                    autocast_dtype=autocast_dtype,
                )

                for name, value in metrics.items():
                    writer.add_scalar(f"eval/{name}", value, epoch + 1)
                writer.add_histogram("eval/q_prediction", predictions, epoch + 1)
                writer.add_histogram("eval/q_target", targets, epoch + 1)

                print(
                    f"    eval_loss={metrics['loss']:.6f}"
                    f" | mae={metrics['mae']:.6f}"
                    f" | spearman={metrics['spearman']:.4f}"
                    f" | rank_acc={metrics['pairwise_ranking_accuracy']:.4f}"
                )

                with (run_directory / "latest_eval_metrics.json").open(
                    "w", encoding="utf-8"
                ) as file:
                    json.dump(
                        {"epoch": epoch + 1, "global_step": global_step, **metrics},
                        file,
                        indent=2,
                    )

                if metrics["loss"] < best_eval_loss:
                    best_eval_loss = metrics["loss"]
                    payload = build_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        epoch,
                        global_step,
                        best_eval_loss,
                        config,
                    )
                    save_checkpoint(best_model_directory / "model.pt", payload)
                    with (best_model_directory / "metrics.json").open(
                        "w", encoding="utf-8"
                    ) as file:
                        json.dump(
                            {"epoch": epoch + 1, "global_step": global_step, **metrics},
                            file,
                            indent=2,
                        )
                    print("    Saved new best model.")

            should_save = (
                (epoch + 1) % save_every_epochs == 0
                or epoch + 1 == epochs
            )
            if should_save:
                payload = build_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    global_step,
                    best_eval_loss,
                    config,
                )
                checkpoint_path = checkpoint_directory / (
                    f"checkpoint-epoch{epoch + 1:04d}-step{global_step:08d}.pt"
                )
                save_checkpoint(checkpoint_path, payload)
                save_checkpoint(checkpoint_directory / "latest.pt", payload)
                rotate_checkpoints(checkpoint_directory, save_total_limit)

        final_payload = build_checkpoint(
            model,
            optimizer,
            scheduler,
            scaler,
            epochs - 1,
            global_step,
            best_eval_loss,
            config,
        )
        save_checkpoint(run_directory / "final_model.pt", final_payload)

        print("=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Final model: {run_directory / 'final_model.pt'}")
        print(f"Best model: {best_model_directory / 'model.pt'}")
        print(f"TensorBoard: {tensorboard_directory}")
    finally:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()
