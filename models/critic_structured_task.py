"""
models/critic.py

Q-only Transformer critic for planner-level offline RL.

It predicts:
    Q(state, task, action)

Tokenization:
    [CLS] + [TASK] + 4 robot tokens + 1 token/object + [ACTION]

The task and action are represented with the same structured quantities:
    skill ID
    target object ID (-1 means no object)
    goal_x
    goal_y
    goal_yaw

The skill embedding, object-ID embedding, and goal projection are shared
between the task and action. Token-type embeddings distinguish their roles.

Attention:
    - No causal mask is used.
    - Every valid token can attend to every other valid token.
    - A padding mask is used only to ignore padded object entries.

Robot state layout (37 values):
    0:7    base position + quaternion
    7:13   base linear velocity + angular velocity
    13:25  joint positions
    25:37  joint velocities

Object physical feature layout (13 values):
    position (3)
    quaternion (4)
    size (3)
    mass (1)
    type (1)
    movable flag (1)

Each object token is:
    224-dimensional physical projection + 32-dimensional object-ID embedding
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn


@dataclass
class CriticConfig:
    # State Transformer
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1

    # Input dimensions
    robot_state_dim: int = 37
    object_feature_dim: int = 13

    # Structured task/action
    num_skills: int = 2
    max_object_id: int = 6

    # Used to normalize task/action goal coordinates
    room_x_limit: float = 1.5
    room_y_limit: float = 2.5

    # Object token split
    object_physical_dim: int = 224
    object_id_embedding_dim: int = 32


@dataclass
class CriticOutput:
    q_value: Tensor  # [B]


class ProjectionMLP(nn.Module):
    """Small MLP used to project one semantic quantity."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        hidden_dim = hidden_dim or output_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class StateActionTransformerCritic(nn.Module):
    """
    Q-only critic over structured task, robot state, scene objects, and action.

    The same learned representations are reused for task and action:
        skill embedding
        object-ID embedding
        goal projection

    Their roles remain distinguishable through token-type embeddings.
    """

    ROBOT_SLICES = {
        "base_pose": slice(0, 7),
        "base_velocity": slice(7, 13),
        "joint_position": slice(13, 25),
        "joint_velocity": slice(25, 37),
    }

    # Token-type IDs
    CLS_TYPE = 0
    TASK_TYPE = 1
    ROBOT_TYPE = 2
    OBJECT_TYPE = 3
    ACTION_TYPE = 4

    def __init__(self, config: CriticConfig) -> None:
        super().__init__()

        self.config = config

        if config.d_model % config.num_heads != 0:
            raise ValueError(
                f"d_model ({config.d_model}) must be divisible by "
                f"num_heads ({config.num_heads})."
            )

        if config.robot_state_dim != 37:
            raise ValueError(
                "This implementation expects the current 37-dimensional "
                "robot state."
            )

        if (
            config.object_physical_dim
            + config.object_id_embedding_dim
            != config.d_model
        ):
            raise ValueError(
                "object_physical_dim + object_id_embedding_dim "
                "must equal d_model."
            )

        if config.num_skills != 2:
            raise ValueError(
                "This implementation expects two skills: "
                "0=navigation and 1=manipulation."
            )

        if config.room_x_limit <= 0 or config.room_y_limit <= 0:
            raise ValueError("Room limits must be positive.")

        # --------------------------------------------------------------
        # Four semantic robot tokens
        # --------------------------------------------------------------
        self.base_pose_projection = ProjectionMLP(
            input_dim=7,
            output_dim=config.d_model,
            dropout=config.dropout,
        )

        self.base_velocity_projection = ProjectionMLP(
            input_dim=6,
            output_dim=config.d_model,
            dropout=config.dropout,
        )

        self.joint_position_projection = ProjectionMLP(
            input_dim=12,
            output_dim=config.d_model,
            dropout=config.dropout,
        )

        self.joint_velocity_projection = ProjectionMLP(
            input_dim=12,
            output_dim=config.d_model,
            dropout=config.dropout,
        )

        # --------------------------------------------------------------
        # Scene-object tokens
        # --------------------------------------------------------------
        self.object_physical_projection = ProjectionMLP(
            input_dim=config.object_feature_dim,
            output_dim=config.object_physical_dim,
            dropout=config.dropout,
        )

        # Shared by scene objects, task target, and action target.
        self.object_id_embedding = nn.Embedding(
            num_embeddings=config.max_object_id + 1,
            embedding_dim=config.object_id_embedding_dim,
            padding_idx=0,
        )

        # --------------------------------------------------------------
        # Shared structured task/action representation
        # --------------------------------------------------------------
        skill_dim = config.d_model // 4
        object_id_dim = config.object_id_embedding_dim
        goal_dim = config.d_model - skill_dim - object_id_dim

        if goal_dim <= 0:
            raise ValueError(
                "The skill and object-ID dimensions leave no dimensions "
                "for the goal representation."
            )

        self.skill_embedding = nn.Embedding(
            num_embeddings=config.num_skills,
            embedding_dim=skill_dim,
        )

        # Shared because task goals and action goals have the same semantics.
        self.goal_projection = ProjectionMLP(
            input_dim=4,
            output_dim=goal_dim,
            hidden_dim=config.d_model,
            dropout=config.dropout,
        )

        # No additional whole-token task/action projection is used.
        # Concatenation already produces exactly d_model dimensions.

        # --------------------------------------------------------------
        # Transformer
        # --------------------------------------------------------------
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, config.d_model)
        )

        self.token_type_embedding = nn.Embedding(
            num_embeddings=5,
            embedding_dim=config.d_model,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.num_layers,
            norm=nn.LayerNorm(config.d_model),
        )

        # --------------------------------------------------------------
        # Scalar Q head
        # --------------------------------------------------------------
        self.q_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 1),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def encode_robot(self, robot_state: Tensor) -> Tensor:
        """Convert [B, 37] into four robot tokens [B, 4, d_model]."""
        if robot_state.ndim != 2:
            raise ValueError(
                "robot_state must have shape [batch, 37]."
            )

        if robot_state.shape[-1] != self.config.robot_state_dim:
            raise ValueError(
                f"Expected robot_state[..., {self.config.robot_state_dim}], "
                f"got {tuple(robot_state.shape)}."
            )

        base_pose = self.base_pose_projection(
            robot_state[:, self.ROBOT_SLICES["base_pose"]]
        )

        base_velocity = self.base_velocity_projection(
            robot_state[:, self.ROBOT_SLICES["base_velocity"]]
        )

        joint_position = self.joint_position_projection(
            robot_state[:, self.ROBOT_SLICES["joint_position"]]
        )

        joint_velocity = self.joint_velocity_projection(
            robot_state[:, self.ROBOT_SLICES["joint_velocity"]]
        )

        return torch.stack(
            [
                base_pose,
                base_velocity,
                joint_position,
                joint_velocity,
            ],
            dim=1,
        )

    def encode_objects(
        self,
        objects: Tensor,
        object_ids: Tensor,
        object_mask: Tensor,
    ) -> Tensor:
        """Build one [physical(224) || ID(32)] token per object slot."""
        if objects.ndim != 3:
            raise ValueError(
                "objects must have shape [batch, num_objects, "
                "object_feature_dim]."
            )

        if objects.shape[-1] != self.config.object_feature_dim:
            raise ValueError(
                f"Expected {self.config.object_feature_dim} object features, "
                f"got {objects.shape[-1]}."
            )

        if object_ids.shape != objects.shape[:2]:
            raise ValueError(
                "object_ids must have shape [batch, num_objects]."
            )

        if object_mask.shape != objects.shape[:2]:
            raise ValueError(
                "object_mask must have shape [batch, num_objects]."
            )

        object_ids = object_ids.long()
        object_mask = object_mask.bool()

        if torch.any(object_ids < 0):
            raise ValueError("Scene object IDs cannot be negative.")

        if torch.any(object_ids > self.config.max_object_id):
            raise ValueError("A scene object ID exceeds max_object_id.")

        if torch.any(object_mask & (object_ids < 1)):
            raise ValueError(
                "Valid scene objects must have IDs >= 1. "
                "ID 0 is reserved for padded slots."
            )

        if torch.any((~object_mask) & (object_ids != 0)):
            raise ValueError(
                "Padded object slots must have object ID 0."
            )

        physical_part = self.object_physical_projection(
            objects
        )  # [B, N, 224]

        id_part = self.object_id_embedding(
            object_ids
        )  # [B, N, 32]

        return torch.cat(
            [physical_part, id_part],
            dim=-1,
        )  # [B, N, 256]

    def _encode_structured_command(
        self,
        skill: Tensor,
        object_id: Tensor,
        goal: Tensor,
        scene_object_ids: Tensor,
        object_mask: Tensor,
        command_name: str,
    ) -> Tensor:
        """
        Encode one structured task or action token.

        Conventions:
            skill == 0: navigation, object_id must be -1
            skill == 1: manipulation, object_id must be >= 1 and present

        The returned token is:
            [shared skill embedding || shared object-ID embedding ||
             shared goal projection]
        """
        if goal.ndim != 2 or goal.shape[-1] != 3:
            raise ValueError(
                f"{command_name}_goal must have shape [batch, 3]."
            )

        if scene_object_ids.ndim != 2:
            raise ValueError(
                "scene_object_ids must have shape [batch, num_objects]."
            )

        if object_mask.shape != scene_object_ids.shape:
            raise ValueError(
                "object_mask must have the same shape as scene_object_ids."
            )

        skill = skill.long()
        object_id = object_id.long()
        scene_object_ids = scene_object_ids.long()
        object_mask = object_mask.bool()

        if skill.ndim != 1:
            raise ValueError(
                f"{command_name}_skill must have shape [batch]."
            )

        if object_id.ndim != 1:
            raise ValueError(
                f"{command_name}_object_id must have shape [batch]."
            )

        batch_size = skill.shape[0]

        if (
            object_id.shape[0] != batch_size
            or goal.shape[0] != batch_size
            or scene_object_ids.shape[0] != batch_size
        ):
            raise ValueError(
                f"All {command_name} and scene inputs must have the same "
                "batch size."
            )

        navigation_mask = skill == 0
        manipulation_mask = skill == 1

        if torch.any(~(navigation_mask | manipulation_mask)):
            raise ValueError(
                f"{command_name}_skill must be 0 (navigation) or "
                "1 (manipulation)."
            )

        if torch.any(navigation_mask & (object_id != -1)):
            bad_indices = torch.nonzero(
                navigation_mask & (object_id != -1),
                as_tuple=False,
            ).flatten().tolist()

            raise ValueError(
                f"Navigation {command_name}s must have "
                f"{command_name}_object_id == -1. "
                f"Invalid batch entries: {bad_indices}"
            )

        if torch.any(manipulation_mask & (object_id < 1)):
            bad_indices = torch.nonzero(
                manipulation_mask & (object_id < 1),
                as_tuple=False,
            ).flatten().tolist()

            raise ValueError(
                f"Manipulation {command_name}s must have "
                f"{command_name}_object_id >= 1. "
                f"Invalid batch entries: {bad_indices}"
            )

        if torch.any(object_id > self.config.max_object_id):
            raise ValueError(
                f"{command_name}_object_id exceeds max_object_id."
            )

        target_matches = (
            scene_object_ids == object_id.unsqueeze(1)
        ) & object_mask

        target_exists = target_matches.any(dim=1)

        if torch.any(manipulation_mask & ~target_exists):
            bad_mask = manipulation_mask & ~target_exists
            bad_indices = torch.nonzero(
                bad_mask,
                as_tuple=False,
            ).flatten().tolist()

            missing_targets = object_id[bad_mask].tolist()

            raise ValueError(
                f"A manipulation {command_name} targets an object that is "
                "not present in the corresponding scene. "
                f"Batch entries: {bad_indices}; "
                f"missing target IDs: {missing_targets}"
            )

        # Navigation's -1 target is represented by embedding index 0.
        embedding_id = torch.where(
            navigation_mask,
            torch.zeros_like(object_id),
            object_id,
        )

        skill_part = self.skill_embedding(skill)
        object_part = self.object_id_embedding(embedding_id)

        goal_x = goal[:, 0] / self.config.room_x_limit
        goal_y = goal[:, 1] / self.config.room_y_limit
        goal_yaw = goal[:, 2]

        goal_features = torch.stack(
            [
                goal_x,
                goal_y,
                torch.sin(goal_yaw),
                torch.cos(goal_yaw),
            ],
            dim=-1,
        )

        goal_part = self.goal_projection(goal_features)

        command_token = torch.cat(
            [skill_part, object_part, goal_part],
            dim=-1,
        )

        if command_token.shape[-1] != self.config.d_model:
            raise RuntimeError(
                f"Structured {command_name} token has dimension "
                f"{command_token.shape[-1]}, expected "
                f"{self.config.d_model}."
            )

        return command_token

    def encode_task(
        self,
        task_skill: Tensor,
        task_object_id: Tensor,
        task_goal: Tensor,
        scene_object_ids: Tensor,
        object_mask: Tensor,
    ) -> Tensor:
        """Build one structured task token [B, d_model]."""
        return self._encode_structured_command(
            skill=task_skill,
            object_id=task_object_id,
            goal=task_goal,
            scene_object_ids=scene_object_ids,
            object_mask=object_mask,
            command_name="task",
        )

    def encode_action(
        self,
        action_skill: Tensor,
        action_object_id: Tensor,
        action_goal: Tensor,
        scene_object_ids: Tensor,
        object_mask: Tensor,
    ) -> Tensor:
        """Build one structured action token [B, d_model]."""
        return self._encode_structured_command(
            skill=action_skill,
            object_id=action_object_id,
            goal=action_goal,
            scene_object_ids=scene_object_ids,
            object_mask=object_mask,
            command_name="action",
        )

    def build_tokens(
        self,
        task_token: Tensor,
        robot_tokens: Tensor,
        object_tokens: Tensor,
        object_mask: Tensor,
        action_token: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Construct:
            [CLS] + [TASK] + 4 robot tokens + object tokens + [ACTION]

        There is no causal mask. The returned padding mask ignores only
        padded object slots.
        """
        batch_size = task_token.shape[0]
        device = task_token.device

        if task_token.ndim != 2:
            raise ValueError(
                "task_token must have shape [batch, d_model]."
            )

        if action_token.ndim != 2:
            raise ValueError(
                "action_token must have shape [batch, d_model]."
            )

        if object_mask.shape != object_tokens.shape[:2]:
            raise ValueError(
                "object_mask must have shape [batch, num_objects]."
            )

        cls = self.cls_token.expand(batch_size, -1, -1)

        token_blocks = [
            cls,
            task_token.unsqueeze(1),
            robot_tokens,
            object_tokens,
            action_token.unsqueeze(1),
        ]

        type_id_blocks = [
            torch.full(
                (1,), self.CLS_TYPE, dtype=torch.long, device=device
            ),
            torch.full(
                (1,), self.TASK_TYPE, dtype=torch.long, device=device
            ),
            torch.full(
                (robot_tokens.shape[1],),
                self.ROBOT_TYPE,
                dtype=torch.long,
                device=device,
            ),
            torch.full(
                (object_tokens.shape[1],),
                self.OBJECT_TYPE,
                dtype=torch.long,
                device=device,
            ),
            torch.full(
                (1,), self.ACTION_TYPE, dtype=torch.long, device=device
            ),
        ]

        padding_blocks = [
            torch.zeros(batch_size, 1, dtype=torch.bool, device=device),
            torch.zeros(batch_size, 1, dtype=torch.bool, device=device),
            torch.zeros(
                batch_size,
                robot_tokens.shape[1],
                dtype=torch.bool,
                device=device,
            ),
            ~object_mask.bool(),
            torch.zeros(batch_size, 1, dtype=torch.bool, device=device),
        ]

        tokens = torch.cat(token_blocks, dim=1)
        type_ids = torch.cat(type_id_blocks, dim=0)
        padding_mask = torch.cat(padding_blocks, dim=1)

        tokens = (
            tokens
            + self.token_type_embedding(type_ids).unsqueeze(0)
        )

        return tokens, padding_mask

    def run_transformer(
        self,
        tokens: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        """Run bidirectional attention and return the final CLS state."""
        encoded = self.transformer(
            src=tokens,
            src_key_padding_mask=padding_mask,
        )

        return encoded[:, 0]

    def forward(
        self,
        task_skill: Tensor,
        task_object_id: Tensor,
        task_goal: Tensor,
        robot_state: Tensor,
        objects: Tensor,
        object_ids: Tensor,
        object_mask: Tensor,
        action_skill: Tensor,
        action_object_id: Tensor,
        action_goal: Tensor,
    ) -> CriticOutput:
        robot_tokens = self.encode_robot(robot_state)

        object_tokens = self.encode_objects(
            objects=objects,
            object_ids=object_ids,
            object_mask=object_mask,
        )

        task_token = self.encode_task(
            task_skill=task_skill,
            task_object_id=task_object_id,
            task_goal=task_goal,
            scene_object_ids=object_ids,
            object_mask=object_mask,
        )

        action_token = self.encode_action(
            action_skill=action_skill,
            action_object_id=action_object_id,
            action_goal=action_goal,
            scene_object_ids=object_ids,
            object_mask=object_mask,
        )

        q_tokens, q_padding_mask = self.build_tokens(
            task_token=task_token,
            robot_tokens=robot_tokens,
            object_tokens=object_tokens,
            object_mask=object_mask,
            action_token=action_token,
        )

        q_representation = self.run_transformer(
            tokens=q_tokens,
            padding_mask=q_padding_mask,
        )

        q_value = self.q_head(
            q_representation
        ).squeeze(-1)

        return CriticOutput(q_value=q_value)


if __name__ == "__main__":
    # --------------------------------------------------------------
    # Basic shape and task/action token tests
    # --------------------------------------------------------------
    config = CriticConfig()
    critic = StateActionTransformerCritic(config)
    critic.eval()

    batch_size = 3
    max_objects = 6

    dummy_batch = {
        "task_skill": torch.tensor(
            [0, 1, 1],
            dtype=torch.long,
        ),
        "task_object_id": torch.tensor(
            [-1, 3, 5],
            dtype=torch.long,
        ),
        "task_goal": torch.tensor(
            [
                [0.8, -1.0, 0.5],
                [-0.4, 1.2, -1.3],
                [1.0, 2.0, 2.7],
            ],
            dtype=torch.float32,
        ),
        "robot_state": torch.randn(
            batch_size,
            config.robot_state_dim,
        ),
        "objects": torch.randn(
            batch_size,
            max_objects,
            config.object_feature_dim,
        ),
        "object_ids": torch.tensor(
            [
                [1, 2, 3, 0, 0, 0],
                [1, 2, 3, 4, 0, 0],
                [1, 2, 3, 4, 5, 6],
            ],
            dtype=torch.long,
        ),
        "object_mask": torch.tensor(
            [
                [True, True, True, False, False, False],
                [True, True, True, True, False, False],
                [True, True, True, True, True, True],
            ],
            dtype=torch.bool,
        ),
        "action_skill": torch.tensor(
            [0, 1, 0],
            dtype=torch.long,
        ),
        "action_object_id": torch.tensor(
            [-1, 2, -1],
            dtype=torch.long,
        ),
        "action_goal": torch.tensor(
            [
                [-1.0, 0.5, 0.2],
                [0.7, -1.5, -2.8],
                [1.2, 2.0, 3.0],
            ],
            dtype=torch.float32,
        ),
    }

    # --------------------------------------------------------------
    # Test 1: full forward pass
    # --------------------------------------------------------------
    with torch.no_grad():
        output = critic(**dummy_batch)

    print("Q shape:", output.q_value.shape)
    print("Q values:", output.q_value)

    assert output.q_value.shape == (batch_size,)
    assert torch.isfinite(output.q_value).all()

    # --------------------------------------------------------------
    # Test 2:
    # Identical task and action content should produce identical
    # raw structured tokens.
    # --------------------------------------------------------------
    same_skill = torch.tensor(
        [1],
        dtype=torch.long,
    )

    same_object_id = torch.tensor(
        [2],
        dtype=torch.long,
    )

    same_goal = torch.tensor(
        [[0.7, -1.5, -2.8]],
        dtype=torch.float32,
    )

    scene_object_ids = torch.tensor(
        [[1, 2, 3, 0, 0, 0]],
        dtype=torch.long,
    )

    scene_object_mask = torch.tensor(
        [[True, True, True, False, False, False]],
        dtype=torch.bool,
    )

    with torch.no_grad():
        task_token = critic.encode_task(
            task_skill=same_skill,
            task_object_id=same_object_id,
            task_goal=same_goal,
            scene_object_ids=scene_object_ids,
            object_mask=scene_object_mask,
        )

        action_token = critic.encode_action(
            action_skill=same_skill,
            action_object_id=same_object_id,
            action_goal=same_goal,
            scene_object_ids=scene_object_ids,
            object_mask=scene_object_mask,
        )

    raw_tokens_match = torch.allclose(
        task_token,
        action_token,
    )

    print(
        "Identical task/action raw tokens match:",
        raw_tokens_match,
    )

    assert raw_tokens_match, (
        "Identical task and action content should produce "
        "identical raw structured tokens."
    )

    # --------------------------------------------------------------
    # Test 3:
    # After adding token-type embeddings, task and action should
    # differ because they have different semantic roles.
    # --------------------------------------------------------------
    task_token_with_type = (
        task_token
        + critic.token_type_embedding.weight[
            critic.TASK_TYPE
        ].unsqueeze(0)
    )

    action_token_with_type = (
        action_token
        + critic.token_type_embedding.weight[
            critic.ACTION_TYPE
        ].unsqueeze(0)
    )

    typed_tokens_match = torch.allclose(
        task_token_with_type,
        action_token_with_type,
    )

    print(
        "Task/action tokens match after type embeddings:",
        typed_tokens_match,
    )

    assert not typed_tokens_match, (
        "Task and action tokens should differ after their "
        "different token-type embeddings are added."
    )

    # Their difference should come exactly from the type embeddings.
    expected_difference = (
        critic.token_type_embedding.weight[
            critic.TASK_TYPE
        ]
        - critic.token_type_embedding.weight[
            critic.ACTION_TYPE
        ]
    )

    actual_difference = (
        task_token_with_type
        - action_token_with_type
    ).squeeze(0)

    torch.testing.assert_close(
        actual_difference,
        expected_difference,
    )

    print("All basic critic tests passed.")