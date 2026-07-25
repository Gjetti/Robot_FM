from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# Configuration
# =============================================================================

FRAME_REFLECTION = np.diag([1.0, -1.0, 1.0])

JOINT_POSITION_COLUMNS = [
    f"joint_pos_wrt_default_{index}"
    for index in range(12)
]

JOINT_VELOCITY_COLUMNS = [
    f"joint_vel_{index}"
    for index in range(12)
]


@dataclass
class CriticDataConfig:
    # Planner-level rollout rows.
    row_csv: str

    # One row per planner segment, including reward_total.
    segment_csv: str

    # Original synthetic dataset containing object_1, object_2, ...
    # strings from which the movable flags are recovered.
    base_csv: str

    max_objects: int = 6
    gamma: float = 0.99


# =============================================================================
# Generic validation and ID helpers
# =============================================================================

def require_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    frame_name: str,
) -> None:
    missing = [
        column
        for column in columns
        if column not in frame.columns
    ]

    if missing:
        raise KeyError(
            f"{frame_name} is missing required columns: {missing}"
        )


def normalize_trajectory_id(value: Any) -> str:
    """
    Normalize IDs so rollout and base CSV trajectory IDs match reliably.

    Examples:
        12      -> "12"
        12.0    -> "12"
        "12.0"  -> "12"
        "trajA" -> "trajA"
    """
    if pd.isna(value):
        raise ValueError("trajectory_id cannot be missing.")

    if isinstance(value, (int, np.integer)):
        return str(int(value))

    if isinstance(value, (float, np.floating)):
        if float(value).is_integer():
            return str(int(value))

        return str(value)

    text = str(value).strip()

    if re.fullmatch(r"[-+]?\d+\.0+", text):
        return text.split(".", maxsplit=1)[0]

    return text


def coerce_boolean_series(
    series: pd.Series,
    column_name: str,
) -> pd.Series:
    """
    Convert common CSV boolean representations to a real bool Series.

    This avoids the problem where bool("False") evaluates to True.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )

    if normalized.isna().any():
        bad_values = (
            series[normalized.isna()]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            f"Could not parse {column_name!r} as boolean. "
            f"Unexpected values: {bad_values[:10]}"
        )

    return normalized.astype(bool)


# =============================================================================
# Coordinate-frame transformations
# =============================================================================

def transform_position(
    x: float,
    y: float,
    z: float,
) -> np.ndarray:
    """Simulator position -> laboratory position."""
    return np.asarray(
        [x, -y, z],
        dtype=np.float32,
    )


def transform_linear_velocity(
    vx: float,
    vy: float,
    vz: float,
) -> np.ndarray:
    """Transform an ordinary world-frame vector."""
    return np.asarray(
        [vx, -vy, vz],
        dtype=np.float32,
    )


def transform_angular_velocity(
    wx: float,
    wy: float,
    wz: float,
) -> np.ndarray:
    """
    Transform an axial vector under the y-axis reflection.

    omega_lab = det(S) * S * omega_sim
              = [-wx, wy, -wz]
    """
    return np.asarray(
        [-wx, wy, -wz],
        dtype=np.float32,
    )


def transform_quaternion_wxyz(
    w: float,
    x: float,
    y: float,
    z: float,
) -> np.ndarray:
    """
    Simulator orientation -> laboratory orientation.

    R_lab = S @ R_sim @ S

    Both input and output use quaternion order:
        [w, x, y, z]
    """
    quaternion_xyzw = np.asarray(
        [x, y, z, w],
        dtype=np.float64,
    )

    norm = np.linalg.norm(quaternion_xyzw)

    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError(
            f"Invalid quaternion: {[w, x, y, z]}"
        )

    quaternion_xyzw /= norm

    rotation_sim = Rotation.from_quat(
        quaternion_xyzw
    ).as_matrix()

    rotation_lab = (
        FRAME_REFLECTION
        @ rotation_sim
        @ FRAME_REFLECTION
    )

    qx, qy, qz, qw = Rotation.from_matrix(
        rotation_lab
    ).as_quat()

    return np.asarray(
        [qw, qx, qy, qz],
        dtype=np.float32,
    )


def transform_rollout_goal(
    goal_x: float,
    goal_y: float,
    goal_yaw: float,
) -> np.ndarray:
    """
    Convert a rollout action goal to laboratory coordinates.

    goal_yaw is already a yaw angle in radians. It is not converted from
    degrees or from another angle representation.

        [x, y, yaw]_lab = [x, -y, -yaw]_sim
    """
    return np.asarray(
        [
            goal_x,
            -goal_y,
            -goal_yaw,
        ],
        dtype=np.float32,
    )


# =============================================================================
# Robot-state construction
# =============================================================================

def build_robot_state(
    row: pd.Series,
    joint_position_columns: Sequence[str],
    joint_velocity_columns: Sequence[str],
) -> np.ndarray:
    position = transform_position(
        float(row["robot_pos_x"]),
        float(row["robot_pos_y"]),
        float(row["robot_pos_z"]),
    )

    quaternion = transform_quaternion_wxyz(
        float(row["robot_quat_w"]),
        float(row["robot_quat_x"]),
        float(row["robot_quat_y"]),
        float(row["robot_quat_z"]),
    )

    # These are world-frame quantities saved by the Isaac Lab rollout.
    linear_velocity = transform_linear_velocity(
        float(row["robot_lin_vel_x"]),
        float(row["robot_lin_vel_y"]),
        float(row["robot_lin_vel_z"]),
    )

    angular_velocity = transform_angular_velocity(
        float(row["robot_ang_vel_x"]),
        float(row["robot_ang_vel_y"]),
        float(row["robot_ang_vel_z"]),
    )

    joint_position = row[
        list(joint_position_columns)
    ].to_numpy(dtype=np.float32)

    joint_velocity = row[
        list(joint_velocity_columns)
    ].to_numpy(dtype=np.float32)

    robot_state = np.concatenate(
        [
            position,          # 3
            quaternion,        # 4
            linear_velocity,   # 3
            angular_velocity,  # 3
            joint_position,    # 12
            joint_velocity,    # 12
        ]
    ).astype(np.float32)

    if robot_state.shape != (37,):
        raise RuntimeError(
            f"Expected robot state shape (37,), "
            f"got {robot_state.shape}."
        )

    if not np.isfinite(robot_state).all():
        raise ValueError(
            "Robot state contains NaN or infinite values."
        )

    return robot_state


# =============================================================================
# Base-CSV lookup and movable flags
# =============================================================================

def build_base_initial_lookup(
    base_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one lookup row per original synthetic trajectory.

    The base CSV is separate from the rollout CSV. It supplies metadata that
    was not saved in the rollout, particularly each object's movable flag.
    """
    require_columns(
        base_df,
        [
            "trajectory_id",
            "state_index",
        ],
        "base CSV",
    )

    base_df = base_df.copy()

    base_df["trajectory_id"] = (
        base_df["trajectory_id"]
        .map(normalize_trajectory_id)
    )

    state_index = pd.to_numeric(
        base_df["state_index"],
        errors="raise",
    )

    base_initial = base_df.loc[
        state_index == 0
    ].copy()

    duplicated_mask = base_initial[
        "trajectory_id"
    ].duplicated(keep=False)

    if duplicated_mask.any():
        duplicated = (
            base_initial.loc[
                duplicated_mask,
                "trajectory_id",
            ]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            "The base CSV contains multiple state_index == 0 "
            f"rows for trajectories: {duplicated[:10]}"
        )

    return base_initial.set_index(
        "trajectory_id",
        drop=True,
    )


def get_movable_flag(
    base_row: pd.Series,
    object_id: int,
) -> int:
    """
    Extract the movable flag from the matching base-CSV object string.

    Expected examples:
        rect,1,[0.788, 0.413],(x=-0.98,y=-1.32,yaw=1.6)
        circle,0,[0.300],(x=0.20,y=0.50,yaw=0.0)
    """
    column = f"object_{object_id}"

    if column not in base_row.index:
        raise KeyError(
            f"Base CSV is missing column {column!r}."
        )

    description = base_row[column]

    if pd.isna(description):
        raise ValueError(
            f"Object {object_id} is present in the rollout "
            "but missing from the matching base trajectory."
        )

    parts = str(description).split(",")

    if len(parts) < 2:
        raise ValueError(
            f"Could not parse movable flag from "
            f"{column}={description!r}."
        )

    try:
        movable = int(parts[1].strip())
    except ValueError as error:
        raise ValueError(
            f"Invalid movable flag in "
            f"{column}={description!r}."
        ) from error

    if movable not in (0, 1):
        raise ValueError(
            f"Movable flag must be 0 or 1, got {movable} "
            f"for object {object_id}."
        )

    return movable


# =============================================================================
# Object-state construction
# =============================================================================

def object_prefix_and_type(
    row: pd.Series,
    object_id: int,
) -> tuple[str, int] | None:
    """
    Return the active rollout prefix and object type.

    Type IDs:
        0 = rectangle
        1 = circle
    """
    rectangle_prefix = f"obj{object_id}_rect"
    circle_prefix = f"obj{object_id}_circle"

    rectangle_present = pd.notna(
        row.get(
            f"{rectangle_prefix}_pos_x",
            np.nan,
        )
    )

    circle_present = pd.notna(
        row.get(
            f"{circle_prefix}_pos_x",
            np.nan,
        )
    )

    if rectangle_present and circle_present:
        raise ValueError(
            f"Object {object_id} is simultaneously stored as "
            "a rectangle and a circle."
        )

    if rectangle_present:
        return rectangle_prefix, 0

    if circle_present:
        return circle_prefix, 1

    return None


def get_base_object_ids(
    base_row: pd.Series,
) -> list[int]:
    """Return all non-empty object IDs stored in one base-CSV row."""
    object_ids: list[int] = []

    for column, value in base_row.items():
        match = re.fullmatch(
            r"object_(\d+)",
            str(column),
        )

        if (
            match is not None
            and pd.notna(value)
        ):
            object_ids.append(
                int(match.group(1))
            )

    return sorted(object_ids)


def validate_scene_against_base(
    row: pd.Series,
    base_row: pd.Series,
    max_objects: int,
) -> None:
    """
    Confirm that the rollout scene and original synthetic scene have the
    same object IDs.
    """
    rollout_object_ids = [
        object_id
        for object_id in range(
            1,
            max_objects + 1,
        )
        if object_prefix_and_type(
            row=row,
            object_id=object_id,
        ) is not None
    ]

    base_object_ids = get_base_object_ids(
        base_row
    )

    unsupported_base_ids = [
        object_id
        for object_id in base_object_ids
        if object_id > max_objects
    ]

    if unsupported_base_ids:
        raise RuntimeError(
            "The base trajectory contains object IDs beyond "
            f"max_objects={max_objects}: {unsupported_base_ids}"
        )

    if rollout_object_ids != base_object_ids:
        raise RuntimeError(
            "Object mismatch between rollout and base CSV. "
            f"trajectory_id={row['trajectory_id']!r}, "
            f"episode_id={row['episode_id']!r}, "
            f"rollout objects={rollout_object_ids}, "
            f"base objects={base_object_ids}."
        )


def build_object_feature(
    row: pd.Series,
    prefix: str,
    object_type: int,
    movable: int,
) -> np.ndarray:
    position = transform_position(
        float(row[f"{prefix}_pos_x"]),
        float(row[f"{prefix}_pos_y"]),
        float(row[f"{prefix}_pos_z"]),
    )

    quaternion = transform_quaternion_wxyz(
        float(row[f"{prefix}_quat_w"]),
        float(row[f"{prefix}_quat_x"]),
        float(row[f"{prefix}_quat_y"]),
        float(row[f"{prefix}_quat_z"]),
    )

    size = np.asarray(
        [
            float(row[f"{prefix}_size_x"]),
            float(row[f"{prefix}_size_y"]),
            float(row[f"{prefix}_size_z"]),
        ],
        dtype=np.float32,
    )

    mass = float(
        row[f"{prefix}_mass"]
    )

    feature = np.concatenate(
        [
            position,                     # 3
            quaternion,                   # 4
            size,                         # 3
            np.asarray([mass]),           # 1
            np.asarray([object_type]),    # 1
            np.asarray([movable]),        # 1
        ]
    ).astype(np.float32)

    if feature.shape != (13,):
        raise RuntimeError(
            f"Expected 13 object features, got "
            f"{feature.shape} for {prefix}."
        )

    if not np.isfinite(feature).all():
        raise ValueError(
            f"Object feature contains NaN or infinite values "
            f"for {prefix}."
        )

    return feature


def build_scene_objects(
    row: pd.Series,
    base_row: pd.Series,
    max_objects: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct padded object features, IDs, and validity masks.

    The movable flag is always read from base_row, which belongs to the
    separate original synthetic CSV.
    """
    objects = np.zeros(
        (max_objects, 13),
        dtype=np.float32,
    )

    object_ids = np.zeros(
        max_objects,
        dtype=np.int64,
    )

    object_mask = np.zeros(
        max_objects,
        dtype=np.bool_,
    )

    slot = 0

    for object_id in range(
        1,
        max_objects + 1,
    ):
        result = object_prefix_and_type(
            row=row,
            object_id=object_id,
        )

        if result is None:
            continue

        prefix, object_type = result

        # This value comes from the separate base CSV.
        movable = get_movable_flag(
            base_row=base_row,
            object_id=object_id,
        )

        objects[slot] = build_object_feature(
            row=row,
            prefix=prefix,
            object_type=object_type,
            movable=movable,
        )

        object_ids[slot] = object_id
        object_mask[slot] = True

        slot += 1

    return (
        objects,
        object_ids,
        object_mask,
    )


# =============================================================================
# Structured-task parsing
# =============================================================================

NUMBER_PATTERN = (
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:[eE][-+]?\d+)?"
)

NAVIGATION_TASK_PATTERN = re.compile(
    rf"""
    navigate(?:_to|\s+to)
    \s*
    \(?
    \s*x\s*=\s*(?P<x>{NUMBER_PATTERN})
    \s*,\s*y\s*=\s*(?P<y>{NUMBER_PATTERN})
    \s*,\s*yaw\s*=\s*(?P<yaw>{NUMBER_PATTERN})
    \s*\)?
    """,
    re.IGNORECASE | re.VERBOSE,
)

MANIPULATION_TASK_PATTERN = re.compile(
    rf"""
    manipulate(?:_to)?
    \s+object[_\s]?(?P<object_id>\d+)
    \s+to
    \s*
    \(?
    \s*x\s*=\s*(?P<x>{NUMBER_PATTERN})
    \s*,\s*y\s*=\s*(?P<y>{NUMBER_PATTERN})
    \s*,\s*yaw\s*=\s*(?P<yaw>{NUMBER_PATTERN})
    \s*\)?
    """,
    re.IGNORECASE | re.VERBOSE,
)


def parse_structured_task(
    task: str,
) -> tuple[int, int, np.ndarray]:
    """
    Parse task text into:
        task_skill
        task_object_id
        task_goal = [x, y, yaw]

    The task already contains yaw in radians and already follows the
    laboratory convention used in the previous tested pipeline.

    Therefore:
        - no rad2deg conversion is performed;
        - no deg2rad conversion is performed;
        - no additional y or yaw sign flip is performed.
    """
    task = str(task).strip()

    navigation_match = (
        NAVIGATION_TASK_PATTERN.fullmatch(task)
    )

    if navigation_match is not None:
        task_skill = 0
        task_object_id = -1
        match = navigation_match

    else:
        manipulation_match = (
            MANIPULATION_TASK_PATTERN.fullmatch(
                task
            )
        )

        if manipulation_match is None:
            raise ValueError(
                f"Unsupported task format: {task!r}"
            )

        task_skill = 1

        task_object_id = int(
            manipulation_match.group(
                "object_id"
            )
        )

        match = manipulation_match

    task_goal = np.asarray(
        [
            float(match.group("x")),
            float(match.group("y")),
            np.deg2rad(
                float(match.group("yaw"))
            ),
        ],
        dtype=np.float32,
    )

    if not np.isfinite(task_goal).all():
        raise ValueError(
            f"Task goal contains NaN or infinite values: "
            f"{task!r}"
        )

    return (
        task_skill,
        task_object_id,
        task_goal,
    )


# =============================================================================
# Return-to-go targets
# =============================================================================

def add_return_to_go(
    segments: pd.DataFrame,
    gamma: float,
) -> pd.DataFrame:
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(
            f"gamma must be in [0, 1], got {gamma}."
        )

    require_columns(
        segments,
        [
            "episode_id",
            "segment_order",
            "reward_total",
        ],
        "segment CSV",
    )

    segments = segments.copy()

    segments = segments.sort_values(
        [
            "episode_id",
            "segment_order",
        ],
        kind="stable",
    )

    segments["q_target"] = np.nan

    for _, episode_segments in segments.groupby(
        "episode_id",
        sort=False,
    ):
        running_return = 0.0

        for index in reversed(
            episode_segments.index.tolist()
        ):
            reward = float(
                segments.at[
                    index,
                    "reward_total",
                ]
            )

            if not np.isfinite(reward):
                raise ValueError(
                    f"Non-finite reward_total at row "
                    f"{index}."
                )

            running_return = (
                reward
                + gamma * running_return
            )

            segments.at[
                index,
                "q_target",
            ] = running_return

    return segments


# =============================================================================
# Critic dataset
# =============================================================================

class CriticDataset(Dataset):
    def __init__(
        self,
        decision_rows: pd.DataFrame,
        segment_rows: pd.DataFrame,
        base_initial: pd.DataFrame,
        max_objects: int = 6,
        joint_position_columns: Sequence[str] = (
            JOINT_POSITION_COLUMNS
        ),
        joint_velocity_columns: Sequence[str] = (
            JOINT_VELOCITY_COLUMNS
        ),
    ) -> None:
        self.max_objects = max_objects
        self.base_initial = base_initial

        self.joint_position_columns = list(
            joint_position_columns
        )

        self.joint_velocity_columns = list(
            joint_velocity_columns
        )

        if len(self.joint_position_columns) != 12:
            raise ValueError(
                "Exactly 12 joint-position columns are required."
            )

        if len(self.joint_velocity_columns) != 12:
            raise ValueError(
                "Exactly 12 joint-velocity columns are required."
            )

        require_columns(
            decision_rows,
            [
                "trajectory_id",
                "episode_id",
                "segment_id",
                "segment_order",
                "task",
                "skill",
                "action_obj_id",
                "goal_x",
                "goal_y",
                "goal_yaw",
                *self.joint_position_columns,
                *self.joint_velocity_columns,
            ],
            "decision rows",
        )

        require_columns(
            segment_rows,
            [
                "episode_id",
                "segment_id",
                "segment_order",
                "reward_total",
                "q_target",
            ],
            "segment rows",
        )

        merged = decision_rows.merge(
            segment_rows[
                [
                    "episode_id",
                    "segment_id",
                    "segment_order",
                    "reward_total",
                    "q_target",
                ]
            ],
            on=[
                "episode_id",
                "segment_id",
                "segment_order",
            ],
            how="inner",
            validate="one_to_one",
        )

        if len(merged) != len(decision_rows):
            raise RuntimeError(
                "Not every decision row matched exactly one "
                "segment target."
            )

        if len(merged) != len(segment_rows):
            raise RuntimeError(
                "Not every segment target matched exactly one "
                "decision row."
            )

        self.samples = merged.reset_index(
            drop=True
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        row = self.samples.iloc[index]

        # trajectory_id is used only to recover metadata from the base CSV.
        # It is never passed to the critic.
        base_trajectory_id = normalize_trajectory_id(
            row["trajectory_id"]
        )

        if (
            base_trajectory_id
            not in self.base_initial.index
        ):
            raise KeyError(
                f"Trajectory {base_trajectory_id!r} "
                "is missing from the base CSV."
            )

        base_row = self.base_initial.loc[
            base_trajectory_id
        ]

        validate_scene_against_base(
            row=row,
            base_row=base_row,
            max_objects=self.max_objects,
        )

        robot_state = build_robot_state(
            row=row,
            joint_position_columns=(
                self.joint_position_columns
            ),
            joint_velocity_columns=(
                self.joint_velocity_columns
            ),
        )

        objects, object_ids, object_mask = (
            build_scene_objects(
                row=row,
                base_row=base_row,
                max_objects=self.max_objects,
            )
        )

        task_skill, task_object_id, task_goal = (
            parse_structured_task(
                task=str(row["task"])
            )
        )

        action_skill = int(
            row["skill"]
        )

        if action_skill not in (0, 1):
            raise ValueError(
                f"Unsupported action skill "
                f"{action_skill}."
            )

        raw_action_object_id = row[
            "action_obj_id"
        ]

        action_object_id = (
            -1
            if pd.isna(raw_action_object_id)
            else int(raw_action_object_id)
        )

        # Navigation actions do not target an object.
        if action_skill == 0:
            action_object_id = -1

        action_goal = transform_rollout_goal(
            goal_x=float(row["goal_x"]),
            goal_y=float(row["goal_y"]),
            goal_yaw=float(row["goal_yaw"]),
        )

        valid_object_ids = set(
            object_ids[
                object_mask
            ].tolist()
        )

        if (
            task_skill == 1
            and task_object_id
            not in valid_object_ids
        ):
            raise ValueError(
                f"Task targets missing object "
                f"{task_object_id} in episode "
                f"{row['episode_id']!r}."
            )

        if (
            action_skill == 1
            and action_object_id
            not in valid_object_ids
        ):
            raise ValueError(
                f"Action targets missing object "
                f"{action_object_id} in episode "
                f"{row['episode_id']!r}."
            )

        return {
            # -------------------------------------------------------------
            # Model inputs
            # -------------------------------------------------------------
            "task_skill": torch.tensor(
                task_skill,
                dtype=torch.long,
            ),

            "task_object_id": torch.tensor(
                task_object_id,
                dtype=torch.long,
            ),

            "task_goal": torch.from_numpy(
                task_goal
            ),

            "robot_state": torch.from_numpy(
                robot_state
            ),

            "objects": torch.from_numpy(
                objects
            ),

            "object_ids": torch.from_numpy(
                object_ids
            ),

            "object_mask": torch.from_numpy(
                object_mask
            ),

            "action_skill": torch.tensor(
                action_skill,
                dtype=torch.long,
            ),

            "action_object_id": torch.tensor(
                action_object_id,
                dtype=torch.long,
            ),

            "action_goal": torch.from_numpy(
                action_goal
            ),

            # -------------------------------------------------------------
            # Training target
            # -------------------------------------------------------------
            "q_target": torch.tensor(
                float(row["q_target"]),
                dtype=torch.float32,
            ),

            # -------------------------------------------------------------
            # Metadata only; never passed to the critic
            # -------------------------------------------------------------
            "episode_id": str(
                row["episode_id"]
            ),

            "trajectory_id": (
                base_trajectory_id
            ),

            "segment_id": row[
                "segment_id"
            ],
        }


# =============================================================================
# Batch collation
# =============================================================================

class CriticCollator:
    """
    Keep model inputs, target, and metadata structurally separate.

    This prevents episode, trajectory, or segment identifiers from
    accidentally being passed into the critic.
    """

    MODEL_INPUT_FIELDS = (
        "task_skill",
        "task_object_id",
        "task_goal",
        "robot_state",
        "objects",
        "object_ids",
        "object_mask",
        "action_skill",
        "action_object_id",
        "action_goal",
    )

    def __call__(
        self,
        samples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        model_inputs = {
            field: torch.stack(
                [
                    sample[field]
                    for sample in samples
                ]
            )
            for field in self.MODEL_INPUT_FIELDS
        }

        q_target = torch.stack(
            [
                sample["q_target"]
                for sample in samples
            ]
        )

        metadata = {
            "episode_id": [
                sample["episode_id"]
                for sample in samples
            ],

            "trajectory_id": [
                sample["trajectory_id"]
                for sample in samples
            ],

            "segment_id": [
                sample["segment_id"]
                for sample in samples
            ],
        }

        return {
            "model_inputs": model_inputs,
            "q_target": q_target,
            "metadata": metadata,
        }


# =============================================================================
# Episode-level train/evaluation split
# =============================================================================

def split_by_episode(
    decision_rows: pd.DataFrame,
    eval_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < eval_ratio < 1.0:
        raise ValueError(
            "eval_ratio must be strictly between 0 and 1."
        )

    episode_ids = (
        decision_rows["episode_id"]
        .drop_duplicates()
        .to_numpy()
    )

    if len(episode_ids) < 2:
        raise ValueError(
            "At least two episodes are required for "
            "a train/evaluation split."
        )

    rng = np.random.default_rng(seed)
    rng.shuffle(episode_ids)

    number_eval = int(
        round(
            len(episode_ids)
            * eval_ratio
        )
    )

    number_eval = min(
        max(number_eval, 1),
        len(episode_ids) - 1,
    )

    eval_episode_ids = set(
        episode_ids[:number_eval]
    )

    eval_rows = decision_rows[
        decision_rows["episode_id"].isin(
            eval_episode_ids
        )
    ].copy()

    train_rows = decision_rows[
        ~decision_rows["episode_id"].isin(
            eval_episode_ids
        )
    ].copy()

    return (
        train_rows,
        eval_rows,
    )


# =============================================================================
# DataLoader factory
# =============================================================================

def create_critic_dataloaders(
    config: CriticDataConfig,
    batch_size: int,
    eval_ratio: float = 0.05,
    seed: int = 42,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:

    # -------------------------------------------------------------------------
    # Load the three separate CSV sources
    # -------------------------------------------------------------------------

    row_df = pd.read_csv(
        config.row_csv,
        low_memory=False,
    )

    segment_df = pd.read_csv(
        config.segment_csv,
        low_memory=False,
    )

    # This is the original synthetic CSV used to recover movable flags.
    base_df = pd.read_csv(
        config.base_csv,
        low_memory=False,
    )

    # -------------------------------------------------------------------------
    # Validate and normalize identifiers
    # -------------------------------------------------------------------------

    require_columns(
        row_df,
        [
            "trajectory_id",
            "episode_id",
            "segment_id",
            "segment_order",
            "is_decision_row",
        ],
        "rollout row CSV",
    )

    require_columns(
        segment_df,
        [
            "episode_id",
            "segment_id",
            "segment_order",
            "reward_total",
        ],
        "segment CSV",
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

    # -------------------------------------------------------------------------
    # Keep one row per planner decision and compute return-to-go
    # -------------------------------------------------------------------------

    decision_mask = coerce_boolean_series(
        row_df["is_decision_row"],
        column_name="is_decision_row",
    )

    decision_rows = row_df.loc[
        decision_mask
    ].copy()

    segment_df = add_return_to_go(
        segments=segment_df,
        gamma=config.gamma,
    )

    # -------------------------------------------------------------------------
    # Build the base trajectory lookup for movable flags
    # -------------------------------------------------------------------------

    base_initial = build_base_initial_lookup(
        base_df
    )

    missing_base_trajectories = sorted(
        set(decision_rows["trajectory_id"])
        - set(base_initial.index)
    )

    if missing_base_trajectories:
        raise KeyError(
            "Some rollout trajectories are absent from the base CSV: "
            f"{missing_base_trajectories[:10]}"
        )

    # -------------------------------------------------------------------------
    # Split complete episodes to avoid train/evaluation leakage
    # -------------------------------------------------------------------------

    train_rows, eval_rows = split_by_episode(
        decision_rows=decision_rows,
        eval_ratio=eval_ratio,
        seed=seed,
    )

    train_episode_ids = set(
        train_rows["episode_id"]
    )

    eval_episode_ids = set(
        eval_rows["episode_id"]
    )

    train_segments = segment_df[
        segment_df["episode_id"].isin(
            train_episode_ids
        )
    ].copy()

    eval_segments = segment_df[
        segment_df["episode_id"].isin(
            eval_episode_ids
        )
    ].copy()

    # -------------------------------------------------------------------------
    # Construct datasets and DataLoaders
    # -------------------------------------------------------------------------

    train_dataset = CriticDataset(
        decision_rows=train_rows,
        segment_rows=train_segments,
        base_initial=base_initial,
        max_objects=config.max_objects,
    )

    eval_dataset = CriticDataset(
        decision_rows=eval_rows,
        segment_rows=eval_segments,
        base_initial=base_initial,
        max_objects=config.max_objects,
    )

    collator = CriticCollator()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
        persistent_workers=(
            num_workers > 0
        ),
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
        persistent_workers=(
            num_workers > 0
        ),
    )

    return (
        train_loader,
        eval_loader,
    )


"""
batch["model_inputs"]   # Passed to critic(**model_inputs)
batch["q_target"]       # Regression target
batch["metadata"]       # Never passed to the critic
"""

if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    # Replace these paths with the actual CSV paths.
    data_config = CriticDataConfig(
        row_csv="/home/georges/BoxPushingProject/source/georges_ext/georges_ext/tasks/locomotion/georges_v0/all_rollouts_50hz_july17_downsampled_5hz_july17_with_rewards.csv",
        segment_csv="/home/georges/BoxPushingProject/source/georges_ext/georges_ext/tasks/locomotion/georges_v0/all_rollouts_50hz_july17_downsampled_5hz_july17_decision_segments_with_rewards.csv",
        base_csv="/home/georges/Go2SDK-main/example/go2/low_level/collected_demos_for_physics_rollouts_filtered_july6.csv",
        max_objects=6,
        gamma=0.99,
    )

    # num_workers=0 keeps this simple and makes errors easier to inspect.
    train_loader, eval_loader = create_critic_dataloaders(
        config=data_config,
        batch_size=3,
        eval_ratio=0.05,
        seed=42,
        num_workers=0,
    )

    # -------------------------------------------------------------------------
    # Retrieve one batch of three samples
    # -------------------------------------------------------------------------
    batch = next(iter(train_loader))

    model_inputs = batch["model_inputs"]
    q_target = batch["q_target"]
    metadata = batch["metadata"]
    # -------------------------------------------------------------------------
    # Print all decision cycles for every episode represented in this batch
    # -------------------------------------------------------------------------

    # Use dict.fromkeys to remove duplicates while preserving sampled order.
    # This matters when two sampled decisions belong to the same episode.
    sampled_episode_ids = list(
        dict.fromkeys(
            metadata["episode_id"]
        )
    )

    print("\n============================================================")
    print("ALL DECISION CYCLES FOR SAMPLED EPISODES")
    print("============================================================")

    # Reload the real row and segment CSVs for inspection.
    inspection_row_df = pd.read_csv(
        data_config.row_csv,
        low_memory=False,
    )

    inspection_segment_df = pd.read_csv(
        data_config.segment_csv,
        low_memory=False,
    )

    # Normalize episode IDs in the same way as the loader.
    inspection_row_df["episode_id"] = (
        inspection_row_df["episode_id"]
        .astype(str)
    )

    inspection_segment_df["episode_id"] = (
        inspection_segment_df["episode_id"]
        .astype(str)
    )

    # Keep only planner decision rows.
    inspection_decision_mask = coerce_boolean_series(
        inspection_row_df["is_decision_row"],
        column_name="is_decision_row",
    )

    inspection_decision_rows = (
        inspection_row_df.loc[
            inspection_decision_mask
            & inspection_row_df["episode_id"].isin(
                sampled_episode_ids
            )
        ]
        .copy()
    )

    # Compute the Monte Carlo discounted return using the complete
    # segment sequence before filtering to the sampled episodes.
    inspection_segment_df = add_return_to_go(
        segments=inspection_segment_df,
        gamma=data_config.gamma,
    )

    inspection_segment_rows = (
        inspection_segment_df[
            inspection_segment_df["episode_id"].isin(
                sampled_episode_ids
            )
        ]
        .copy()
    )

    # Select the decision-state information from the rollout CSV.
    decision_columns = [
        "trajectory_id",
        "episode_id",
        "segment_id",
        "segment_order",
        "task",
        "skill",
        "action_obj_id",
        "goal_x",
        "goal_y",
        "goal_yaw",
    ]

    # Select rewards and discounted return from the segment CSV.
    target_columns = [
        "episode_id",
        "segment_id",
        "segment_order",
    ]

    # Include available reward components automatically.
    for reward_column in [
        "reward_completion",
        "reward_collision",
        "reward_oob",
        "reward_total",
    ]:
        if reward_column in inspection_segment_rows.columns:
            target_columns.append(
                reward_column
            )

    target_columns.append(
        "q_target"
    )

    all_sampled_decisions = (
        inspection_decision_rows[
            decision_columns
        ]
        .merge(
            inspection_segment_rows[
                target_columns
            ],
            on=[
                "episode_id",
                "segment_id",
                "segment_order",
            ],
            how="left",
            validate="one_to_one",
        )
        .sort_values(
            [
                "episode_id",
                "segment_order",
            ],
            kind="stable",
        )
    )

    # Every sampled decision must have a matching segment reward and return.
    if all_sampled_decisions["q_target"].isna().any():
        unmatched = all_sampled_decisions[
            all_sampled_decisions["q_target"].isna()
        ][
            [
                "episode_id",
                "segment_id",
                "segment_order",
            ]
        ]

        raise RuntimeError(
            "Some sampled decision rows did not match the segment CSV:\n"
            f"{unmatched.to_string(index=False)}"
        )

    # Print each sampled episode separately and in temporal order.
    for episode_id in sampled_episode_ids:
        episode_decisions = (
            all_sampled_decisions[
                all_sampled_decisions["episode_id"]
                == episode_id
            ]
            .sort_values(
                "segment_order",
                kind="stable",
            )
        )

        if episode_decisions.empty:
            raise RuntimeError(
                f"No decision rows found for sampled episode "
                f"{episode_id!r}."
            )

        trajectory_id = episode_decisions[
            "trajectory_id"
        ].iloc[0]

        print("\n------------------------------------------------------------")
        print(
            f"trajectory_id: {trajectory_id} | "
            f"episode_id: {episode_id}"
        )
        print("------------------------------------------------------------")

        columns_to_print = [
            "segment_order",
            "segment_id",
            "skill",
            "action_obj_id",
            "goal_x",
            "goal_y",
            "goal_yaw",
        ]

        for reward_column in [
            "reward_completion",
            "reward_collision",
            "reward_oob",
            "reward_total",
        ]:
            if reward_column in episode_decisions.columns:
                columns_to_print.append(
                    reward_column
                )

        columns_to_print.append(
            "q_target"
        )

        print(
            episode_decisions[
                columns_to_print
            ].to_string(
                index=False
            )
        )

        print(
            "\nTask stored for these decisions:"
        )

        print(
            episode_decisions[
                [
                    "segment_order",
                    "task",
                ]
            ].to_string(
                index=False
            )
        )
    # -------------------------------------------------------------------------
    # Print batch structure
    # -------------------------------------------------------------------------
    print("\n============================================================")
    print("MODEL INPUT SHAPES")
    print("============================================================")

    for name, tensor in model_inputs.items():
        print(
            f"{name:20s} "
            f"shape={tuple(tensor.shape)}, "
            f"dtype={tensor.dtype}"
        )

    print(
        f"{'q_target':20s} "
        f"shape={tuple(q_target.shape)}, "
        f"dtype={q_target.dtype}"
    )

    # -------------------------------------------------------------------------
    # Verify that the loader output matches the critic interface
    # -------------------------------------------------------------------------
    batch_size = 3
    max_objects = data_config.max_objects

    assert model_inputs["task_skill"].shape == (
        batch_size,
    )

    assert model_inputs["task_object_id"].shape == (
        batch_size,
    )

    assert model_inputs["task_goal"].shape == (
        batch_size,
        3,
    )

    assert model_inputs["robot_state"].shape == (
        batch_size,
        37,
    )

    assert model_inputs["objects"].shape == (
        batch_size,
        max_objects,
        13,
    )

    assert model_inputs["object_ids"].shape == (
        batch_size,
        max_objects,
    )

    assert model_inputs["object_mask"].shape == (
        batch_size,
        max_objects,
    )

    assert model_inputs["action_skill"].shape == (
        batch_size,
    )

    assert model_inputs["action_object_id"].shape == (
        batch_size,
    )

    assert model_inputs["action_goal"].shape == (
        batch_size,
        3,
    )

    assert q_target.shape == (
        batch_size,
    )

    # Metadata must not be included among the critic inputs.
    assert "trajectory_id" not in model_inputs
    assert "episode_id" not in model_inputs
    assert "segment_id" not in model_inputs

    # -------------------------------------------------------------------------
    # Print each processed sample
    # -------------------------------------------------------------------------
    print("\n============================================================")
    print("THREE PROCESSED SAMPLES")
    print("============================================================")

    for index in range(batch_size):
        valid_objects = model_inputs[
            "object_mask"
        ][index]

        print(f"\nSample {index + 1}")

        print(
            "  episode_id:",
            metadata["episode_id"][index],
        )

        print(
            "  trajectory_id:",
            metadata["trajectory_id"][index],
        )

        print(
            "  segment_id:",
            metadata["segment_id"][index],
        )

        print(
            "  task_skill:",
            model_inputs[
                "task_skill"
            ][index].item(),
        )

        print(
            "  task_object_id:",
            model_inputs[
                "task_object_id"
            ][index].item(),
        )

        print(
            "  task_goal:",
            model_inputs[
                "task_goal"
            ][index].tolist(),
        )

        print(
            "  action_skill:",
            model_inputs[
                "action_skill"
            ][index].item(),
        )

        print(
            "  action_object_id:",
            model_inputs[
                "action_object_id"
            ][index].item(),
        )

        print(
            "  action_goal:",
            model_inputs[
                "action_goal"
            ][index].tolist(),
        )

        print(
            "  valid object IDs:",
            model_inputs[
                "object_ids"
            ][index][valid_objects].tolist(),
        )

        print(
            "  object mask:",
            model_inputs[
                "object_mask"
            ][index].tolist(),
        )

        print(
            "  movable flags:",
            model_inputs[
                "objects"
            ][index, valid_objects, 12].tolist(),
        )

        print(
            "  object types:",
            model_inputs[
                "objects"
            ][index, valid_objects, 11].tolist(),
        )

        print(
            "  q_target:",
            q_target[index].item(),
        )

    # -------------------------------------------------------------------------
    # Pass the loader output directly into the critic
    # -------------------------------------------------------------------------
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
        
    from models.critic_structured_task import (
        CriticConfig,
        StateActionTransformerCritic,
    )

    critic_config = CriticConfig(
        max_object_id=data_config.max_objects,
        object_feature_dim=13,
    )

    critic = StateActionTransformerCritic(
        critic_config
    )

    critic.eval()

    with torch.no_grad():
        output = critic(**model_inputs)

    # -------------------------------------------------------------------------
    # Display critic output versus training target
    # -------------------------------------------------------------------------
    print("\n============================================================")
    print("CRITIC OUTPUT")
    print("============================================================")

    print("Predicted Q values:", output.q_value)
    print("Expected Q targets:", q_target)

    assert output.q_value.shape == q_target.shape
    assert torch.isfinite(output.q_value).all()
    assert torch.isfinite(q_target).all()

    print("\nLoader output matches the critic input format.")
    print("The batch was successfully passed through the critic.")