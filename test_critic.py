"""
Evaluate a trained structured Transformer critic on complete sampled episodes.

The script:
1. samples NUM_EPISODES complete episode_id values;
2. keeps every planner decision row from each sampled episode;
3. reconstructs the same Monte Carlo return-to-go targets used in training;
4. predicts one Q-value for every decision state/action pair;
5. saves target-versus-prediction results for later ranking analysis.

Run from the Robot_FM repository root:

    python3 test_critic.py
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

from models.critic_structured_task import (
    CriticConfig,
    StateActionTransformerCritic,
)
from robot_fm_data.critic_data import (
    CriticCollator,
    CriticDataset,
    add_return_to_go,
    build_base_initial_lookup,
    coerce_boolean_series,
    normalize_trajectory_id,
)


# =============================================================================
# Configuration
# =============================================================================

# Prefer the best validation checkpoint for the first test.
CHECKPOINT_PATH = Path(
    "checkpoints/critic/"
    "structured_q_critic_lr0.0003_20260723_130123/"
    "best_model/model.pt"
)

ROW_CSV = Path(
    "/home/georges/BoxPushingProject/source/georges_ext/"
    "georges_ext/tasks/locomotion/georges_v0/"
    "all_rollouts_50hz_july17_downsampled_5hz_july17_with_rewards.csv"
)

SEGMENT_CSV = Path(
    "/home/georges/BoxPushingProject/source/georges_ext/"
    "georges_ext/tasks/locomotion/georges_v0/"
    "all_rollouts_50hz_july17_downsampled_5hz_july17_"
    "decision_segments_with_rewards.csv"
)

BASE_CSV = Path(
    "/home/georges/Go2SDK-main/example/go2/low_level/"
    "collected_demos_for_physics_rollouts_filtered_july6.csv"
)

NUM_EPISODES = 500
BATCH_SIZE = 128
NUM_WORKERS = 0
SEED = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Must match the return definition used during training.
GAMMA = 0.99
MAX_OBJECTS = 6

# =============================================================================
# Reproducibility
# =============================================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =============================================================================
# Helpers
# =============================================================================

def safe_correlation(
    target: np.ndarray,
    prediction: np.ndarray,
    kind: str,
) -> float:
    if target.size < 2:
        return float("nan")

    if np.allclose(target, target[0]):
        return float("nan")

    if np.allclose(prediction, prediction[0]):
        return float("nan")

    if kind == "pearson":
        return float(
            pearsonr(target, prediction).statistic
        )

    if kind == "spearman":
        return float(
            spearmanr(target, prediction).statistic
        )

    raise ValueError(
        f"Unsupported correlation kind: {kind}"
    )



def classify_task_family(
    episode: pd.DataFrame,
) -> str:
    """
    Classify one complete episode into a coarse task family.

    Rules:
    1. If the task text contains "manipulate", it is a manipulation task.
    2. Otherwise, if any segment skill contains "manipulate", it is a
       navigation + manipulation episode.
    3. Otherwise, it is pure navigation.
    """
    episode = episode.sort_values(
        "segment_order",
        kind="stable",
    )

    task_text = " ".join(
        episode["task"]
        .fillna("")
        .astype(str)
        .str.lower()
        .unique()
    )

    segment_skills = (
        episode["skill"]
        .fillna("")
        .astype(str)
        .str.lower()
        .tolist()
    )

    number_decisions = len(episode)

    task_is_manipulation = (
        "manipulate" in task_text
    )

    has_manipulation_segment = any(
        "manipulate" in skill
        for skill in segment_skills
    )

    if task_is_manipulation:
        return "Manipulation task"

    if has_manipulation_segment:
        if number_decisions == 2:
            return (
                "Navigation + manipulation "
                "(2 segments)"
            )

        return (
            "Navigation + manipulation "
            "(>2 segments)"
        )

    return "Pure navigation (1 segment)"

def save_prediction_scatter(
    prediction_df: pd.DataFrame,
    path: Path,
) -> None:
    target = prediction_df["return_to_go"].to_numpy(dtype=float)
    prediction = prediction_df["predicted_q"].to_numpy(dtype=float)

    lower = float(min(target.min(), prediction.min()))
    upper = float(max(target.max(), prediction.max()))
    margin = 0.05 * max(upper - lower, 1.0)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(target, prediction, alpha=0.45, s=18)
    ax.plot(
        [lower - margin, upper + margin],
        [lower - margin, upper + margin],
        linestyle="--",
        linewidth=1.5,
        label="Ideal: predicted Q = return-to-go",
    )
    ax.set_xlabel("Return-to-go target")
    ax.set_ylabel("Predicted Q-value")
    ax.set_title("Predicted Q-value versus return-to-go")
    ax.set_xlim(lower - margin, upper + margin)
    ax.set_ylim(lower - margin, upper + margin)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_residual_plot(
    prediction_df: pd.DataFrame,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(
        prediction_df["return_to_go"],
        prediction_df["prediction_error"],
        alpha=0.45,
        s=18,
    )
    ax.axhline(0.0, linestyle="--", linewidth=1.5)
    ax.set_xlabel("Return-to-go target")
    ax.set_ylabel("Residual: predicted Q - return-to-go")
    ax.set_title("Critic residuals versus return-to-go")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_family_boxplot(
    dataframe: pd.DataFrame,
    value_column: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    family_order = [
        "Pure navigation (1 segment)",
        "Manipulation task",
        "Navigation + manipulation (2 segments)",
        "Navigation + manipulation (>2 segments)",
    ]

    labels: list[str] = []
    values: list[np.ndarray] = []

    for family in family_order:
        family_values = (
            dataframe.loc[
                dataframe["task_family"] == family,
                value_column,
            ]
            .dropna()
            .to_numpy(dtype=float)
        )

        if family_values.size == 0:
            continue

        labels.append(f"{family}\n(n={family_values.size})")
        values.append(family_values)

    if not values:
        print(f"Skipping {path.name}: no task-family values were available.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.boxplot(values, labels=labels, showfliers=True)
    if value_column == "prediction_error":
        ax.axhline(0.0, linestyle="--", linewidth=1.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelrotation=15)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_episode_sequence_pdf(
    prediction_df: pd.DataFrame,
    path: Path,
) -> int:
    """Save one page per multi-decision episode.

    Every page compares the critic prediction and return-to-go at every
    decision state in that episode.
    """
    multi_episode_ids = (
        prediction_df.groupby("episode_id").size()
        .loc[lambda counts: counts > 1]
        .index
        .tolist()
    )

    with PdfPages(path) as pdf:
        for episode_id in multi_episode_ids:
            episode = (
                prediction_df.loc[
                    prediction_df["episode_id"] == episode_id
                ]
                .sort_values("segment_order", kind="stable")
            )

            x = episode["segment_order"].to_numpy(dtype=int)

            fig, ax = plt.subplots(figsize=(9, 6))
            ax.plot(
                x,
                episode["return_to_go"],
                marker="o",
                linewidth=2,
                label="Return-to-go",
            )
            ax.plot(
                x,
                episode["predicted_q"],
                marker="o",
                linewidth=2,
                label="Predicted Q-value",
            )
            ax.set_xticks(x)
            ax.set_xlabel("Decision / segment order")
            ax.set_ylabel("Value")
            ax.set_title(
                f"Episode {episode_id} | "
                f"trajectory {episode['trajectory_id'].iloc[0]} | "
                f"{episode['task_family'].iloc[0]}"
            )
            ax.grid(True, alpha=0.25)
            ax.legend()

            skill_labels = episode["skill"].astype(str).tolist()
            for xi, label in zip(x, skill_labels):
                ax.annotate(
                    label,
                    (xi, episode.loc[
                        episode["segment_order"] == xi,
                        "return_to_go",
                    ].iloc[0]),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=8,
                )

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    return len(multi_episode_ids)

def infer_run_name(
    checkpoint_path: Path,
) -> str:
    """
    Handle either:
        <run>/best_model/model.pt
        <run>/checkpoints/latest.pt
        <run>/final_model.pt
    """
    if checkpoint_path.parent.name in {
        "best_model",
        "checkpoints",
    }:
        return checkpoint_path.parents[1].name

    return checkpoint_path.parent.name


def load_critic(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[
    StateActionTransformerCritic,
    dict[str, Any],
]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if "config" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain the training configuration."
        )

    training_config = checkpoint["config"]

    critic_config = CriticConfig(
        **training_config["model"]
    )

    critic = StateActionTransformerCritic(
        critic_config
    )

    critic.load_state_dict(
        checkpoint["model_state_dict"]
    )

    critic.to(device)
    critic.eval()

    return critic, training_config


# =============================================================================
# Output folder
# =============================================================================

run_name = infer_run_name(
    CHECKPOINT_PATH
)

output_dir = (
    Path("test_results")
    / "critic"
    / run_name
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

predictions_csv = (
    output_dir
    / "q_predictions_500_episodes.csv"
)

episode_summary_csv = (
    output_dir
    / "episode_summary.csv"
)

metrics_json = (
    output_dir
    / "metrics.json"
)

plots_dir = output_dir / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

prediction_scatter_png = plots_dir / "01_predicted_q_vs_return_to_go.png"
residual_plot_png = plots_dir / "02_residuals_vs_return_to_go.png"
signed_error_family_png = plots_dir / "03_signed_error_by_task_family.png"
episode_mae_family_png = plots_dir / "04_episode_mae_by_task_family.png"
episode_sequences_pdf = plots_dir / "05_multisegment_episode_sequences.pdf"


# =============================================================================
# Load and prepare the complete dataset
# =============================================================================

print("Loading CSV files...")

row_df = pd.read_csv(
    ROW_CSV,
    low_memory=False,
)

segment_df = pd.read_csv(
    SEGMENT_CSV,
    low_memory=False,
)

base_df = pd.read_csv(
    BASE_CSV,
    low_memory=False,
)

row_df = row_df.copy()
segment_df = segment_df.copy()

row_df["trajectory_id"] = (
    row_df["trajectory_id"]
    .map(normalize_trajectory_id)
)

row_df["episode_id"] = (
    row_df["episode_id"]
    .astype(str)
)

segment_df["episode_id"] = (
    segment_df["episode_id"]
    .astype(str)
)

decision_mask = coerce_boolean_series(
    row_df["is_decision_row"],
    column_name="is_decision_row",
)

decision_rows = (
    row_df.loc[decision_mask]
    .copy()
)

# Compute returns over complete episodes before sampling.
segment_df = add_return_to_go(
    segments=segment_df,
    gamma=GAMMA,
)


# =============================================================================
# Sample complete episodes
# =============================================================================

available_episode_ids = (
    decision_rows["episode_id"]
    .drop_duplicates()
    .sort_values()
    .to_numpy()
)

if len(available_episode_ids) == 0:
    raise RuntimeError(
        "No episodes with decision rows were found."
    )

number_to_sample = min(
    NUM_EPISODES,
    len(available_episode_ids),
)

rng = np.random.default_rng(SEED)

sampled_episode_ids = rng.choice(
    available_episode_ids,
    size=number_to_sample,
    replace=False,
)

sampled_episode_set = set(
    sampled_episode_ids.tolist()
)

selected_decision_rows = (
    decision_rows[
        decision_rows["episode_id"].isin(
            sampled_episode_set
        )
    ]
    .sort_values(
        [
            "episode_id",
            "segment_order",
        ],
        kind="stable",
    )
    .copy()
)

# Keep exactly the segment targets corresponding to the selected decisions.
selected_keys = selected_decision_rows[
    [
        "episode_id",
        "segment_id",
        "segment_order",
    ]
]

selected_segment_rows = selected_keys.merge(
    segment_df,
    on=[
        "episode_id",
        "segment_id",
        "segment_order",
    ],
    how="left",
    validate="one_to_one",
)

if selected_segment_rows["q_target"].isna().any():
    missing = selected_segment_rows[
        selected_segment_rows["q_target"].isna()
    ][
        [
            "episode_id",
            "segment_id",
            "segment_order",
        ]
    ]

    raise RuntimeError(
        "Some selected decisions did not match a segment target:\n"
        f"{missing.head(20).to_string(index=False)}"
    )

print("\n=== SAMPLED EPISODES ===")
print(f"Available episodes: {len(available_episode_ids)}")
print(f"Sampled episodes:   {number_to_sample}")
print(
    "Decision samples:   "
    f"{len(selected_decision_rows)}"
)


# =============================================================================
# Build the critic dataset
# =============================================================================

base_initial = build_base_initial_lookup(
    base_df
)

selected_trajectory_ids = set(
    selected_decision_rows["trajectory_id"]
)

missing_base_ids = sorted(
    selected_trajectory_ids
    - set(base_initial.index)
)

if missing_base_ids:
    raise KeyError(
        "The base CSV is missing sampled trajectories: "
        f"{missing_base_ids[:20]}"
    )

test_dataset = CriticDataset(
    decision_rows=selected_decision_rows,
    segment_rows=selected_segment_rows,
    base_initial=base_initial,
    max_objects=MAX_OBJECTS,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=CriticCollator(),
    pin_memory=(
        DEVICE == "cuda"
    ),
)


# =============================================================================
# Load trained critic
# =============================================================================

device = torch.device(
    DEVICE
)

critic, checkpoint_config = load_critic(
    checkpoint_path=CHECKPOINT_PATH,
    device=device,
)

checkpoint_gamma = float(
    checkpoint_config["dataset"]["gamma"]
)

checkpoint_max_objects = int(
    checkpoint_config["dataset"]["max_objects"]
)

if not np.isclose(
    checkpoint_gamma,
    GAMMA,
):
    raise ValueError(
        f"Test GAMMA={GAMMA} does not match "
        f"checkpoint gamma={checkpoint_gamma}."
    )

if checkpoint_max_objects != MAX_OBJECTS:
    raise ValueError(
        f"Test MAX_OBJECTS={MAX_OBJECTS} does not match "
        f"checkpoint max_objects={checkpoint_max_objects}."
    )


# =============================================================================
# Critic inference
# =============================================================================

records: list[dict[str, Any]] = []

if device.type == "cuda":
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

start_time = time.time()

with torch.no_grad():
    for batch_index, batch in enumerate(
        test_loader
    ):
        model_inputs = {
            name: tensor.to(
                device=device,
                non_blocking=True,
            )
            for name, tensor
            in batch["model_inputs"].items()
        }

        output = critic(
            **model_inputs
        )

        predicted_q = (
            output.q_value
            .detach()
            .float()
            .cpu()
            .numpy()
        )

        target_q = (
            batch["q_target"]
            .detach()
            .float()
            .cpu()
            .numpy()
        )

        metadata = batch["metadata"]

        for index in range(
            len(predicted_q)
        ):
            records.append(
                {
                    "episode_id": (
                        metadata["episode_id"][index]
                    ),
                    "trajectory_id": (
                        metadata["trajectory_id"][index]
                    ),
                    "segment_id": (
                        metadata["segment_id"][index]
                    ),
                    "predicted_q": float(
                        predicted_q[index]
                    ),
                    "return_to_go": float(
                        target_q[index]
                    ),
                }
            )

        print(
            f"\rProcessed batch "
            f"{batch_index + 1}/{len(test_loader)}",
            end="",
            flush=True,
        )

if device.type == "cuda":
    torch.cuda.synchronize()

elapsed_time = time.time() - start_time

print()


# =============================================================================
# Attach readable decision information
# =============================================================================

prediction_df = pd.DataFrame(
    records
)

decision_information_columns = [
    "episode_id",
    "trajectory_id",
    "segment_id",
    "segment_order",
    "task",
    "skill",
    "action_obj_id",
    "goal_x",
    "goal_y",
    "goal_yaw",
]

for optional_column in [
    "termination_flag",
    "termination_type",
    "reward_completion",
    "reward_collision",
    "reward_oob",
    "reward_total",
]:
    if optional_column in selected_decision_rows.columns:
        decision_information_columns.append(
            optional_column
        )

decision_information = (
    selected_decision_rows[
        decision_information_columns
    ]
    .copy()
)

prediction_df = prediction_df.merge(
    decision_information,
    on=[
        "episode_id",
        "trajectory_id",
        "segment_id",
    ],
    how="left",
    validate="one_to_one",
)

prediction_df["prediction_error"] = (
    prediction_df["predicted_q"]
    - prediction_df["return_to_go"]
)

prediction_df["absolute_error"] = (
    prediction_df["prediction_error"]
    .abs()
)

prediction_df = prediction_df.sort_values(
    [
        "episode_id",
        "segment_order",
    ],
    kind="stable",
).reset_index(
    drop=True
)

expected_decisions = selected_decision_rows[
    ["episode_id", "segment_id", "segment_order"]
].drop_duplicates()

evaluated_decisions = prediction_df[
    ["episode_id", "segment_id", "segment_order"]
].drop_duplicates()

coverage = expected_decisions.merge(
    evaluated_decisions,
    on=["episode_id", "segment_id", "segment_order"],
    how="outer",
    indicator=True,
)

missing_evaluations = coverage.loc[coverage["_merge"] != "both"]
if not missing_evaluations.empty:
    raise RuntimeError(
        "Decision-state coverage check failed:\n"
        f"{missing_evaluations.head(20).to_string(index=False)}"
    )

print(
    "Decision-state coverage check passed: "
    f"{len(evaluated_decisions)}/{len(expected_decisions)} decisions evaluated."
)

prediction_df.to_csv(
    predictions_csv,
    index=False,
)


# =============================================================================
# Overall target-versus-prediction metrics
# =============================================================================

target = prediction_df[
    "return_to_go"
].to_numpy(
    dtype=np.float64
)

prediction = prediction_df[
    "predicted_q"
].to_numpy(
    dtype=np.float64
)

error = prediction - target

overall_metrics = {
    "checkpoint": str(
        CHECKPOINT_PATH
    ),
    "seed": SEED,
    "sampled_episodes": (
        number_to_sample
    ),
    "decision_samples": int(
        len(prediction_df)
    ),
    "gamma": GAMMA,
    "mae": float(
        np.mean(
            np.abs(error)
        )
    ),
    "rmse": float(
        np.sqrt(
            np.mean(
                error ** 2
            )
        )
    ),
    "pearson": safe_correlation(
        target,
        prediction,
        "pearson",
    ),
    "spearman_global": safe_correlation(
        target,
        prediction,
        "spearman",
    ),
    "target_mean": float(
        np.mean(target)
    ),
    "target_std": float(
        np.std(target)
    ),
    "prediction_mean": float(
        np.mean(prediction)
    ),
    "prediction_std": float(
        np.std(prediction)
    ),
    "total_inference_seconds": float(
        elapsed_time
    ),
    "milliseconds_per_decision": float(
        1000.0
        * elapsed_time
        / len(prediction_df)
    ),
}

with metrics_json.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        overall_metrics,
        file,
        indent=2,
    )


# =============================================================================
# Per-episode summary
# =============================================================================

episode_rows: list[dict[str, Any]] = []

for episode_id, episode in prediction_df.groupby(
    "episode_id",
    sort=False,
):
    episode = episode.sort_values(
        "segment_order",
        kind="stable",
    )

    episode_target = episode[
        "return_to_go"
    ].to_numpy(
        dtype=np.float64
    )

    episode_prediction = episode[
        "predicted_q"
    ].to_numpy(
        dtype=np.float64
    )

    episode_error = (
        episode_prediction
        - episode_target
    )

    episode_rows.append(
        {
            "episode_id": episode_id,
            "trajectory_id": (
                episode[
                    "trajectory_id"
                ].iloc[0]
            ),
            "number_decisions": int(
                len(episode)
            ),
            "task": str(episode["task"].iloc[0]),
            "first_return_to_go": float(
                episode_target[0]
            ),
            "first_predicted_q": float(
                episode_prediction[0]
            ),
            "episode_mae": float(
                np.mean(
                    np.abs(
                        episode_error
                    )
                )
            ),
            # Spearman is undefined for one-decision episodes.
            "episode_spearman": (
                safe_correlation(
                    episode_target,
                    episode_prediction,
                    "spearman",
                )
            ),
        }
    )

episode_summary_df = pd.DataFrame(
    episode_rows
)

episode_family_map = {}

for episode_id, episode in prediction_df.groupby(
    "episode_id",
    sort=False,
):
    episode_family_map[episode_id] = (
        classify_task_family(
            episode
        )
    )

episode_summary_df["task_family"] = (
    episode_summary_df["episode_id"]
    .map(episode_family_map)
)

prediction_df = prediction_df.merge(
    episode_summary_df[["episode_id", "task_family"]],
    on="episode_id",
    how="left",
    validate="many_to_one",
)

# Rewrite the detailed CSV so it also contains the task family.
prediction_df.to_csv(
    predictions_csv,
    index=False,
)

episode_summary_df.to_csv(
    episode_summary_csv,
    index=False,
)


# =============================================================================
# Visual evaluation report
# =============================================================================

print("\nGenerating evaluation plots...")

save_prediction_scatter(
    prediction_df=prediction_df,
    path=prediction_scatter_png,
)

save_residual_plot(
    prediction_df=prediction_df,
    path=residual_plot_png,
)

save_family_boxplot(
    dataframe=prediction_df,
    value_column="prediction_error",
    ylabel="Signed error: predicted Q - return-to-go",
    title="Signed critic error by task family",
    path=signed_error_family_png,
)

save_family_boxplot(
    dataframe=episode_summary_df,
    value_column="episode_mae",
    ylabel="Episode mean absolute error",
    title="Episode-level MAE by task family",
    path=episode_mae_family_png,
)

number_episode_pages = save_episode_sequence_pdf(
    prediction_df=prediction_df,
    path=episode_sequences_pdf,
)


# =============================================================================
# Print summary
# =============================================================================

print("\n============================================================")
print("CRITIC TEST RESULTS")
print("============================================================")

print(f"Checkpoint: {CHECKPOINT_PATH}")
print(f"Device: {device}")
print(f"Episodes: {number_to_sample}")
print(f"Decision samples: {len(prediction_df)}")
print(f"MAE: {overall_metrics['mae']:.6f}")
print(f"RMSE: {overall_metrics['rmse']:.6f}")
print(f"Pearson: {overall_metrics['pearson']:.6f}")
print(
    "Global Spearman: "
    f"{overall_metrics['spearman_global']:.6f}"
)

print(
    "Inference time per decision: "
    f"{overall_metrics['milliseconds_per_decision']:.3f} ms"
)

print("\nSaved decision predictions to:")
print(predictions_csv)

print("\nSaved episode summaries to:")
print(episode_summary_csv)

print("\nSaved overall metrics to:")
print(metrics_json)

print("\nSaved visual plots to:")
print(plots_dir)
print(
    "Multi-segment episode pages: "
    f"{number_episode_pages}"
)

print("\nFirst 20 target-versus-prediction rows:")

print(
    prediction_df[
        [
            "episode_id",
            "segment_order",
            "segment_id",
            "skill",
            "action_obj_id",
            "return_to_go",
            "predicted_q",
            "absolute_error",
        ]
    ]
    .head(20)
    .to_string(index=False)
)