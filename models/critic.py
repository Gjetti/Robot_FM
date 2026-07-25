"""
models/critic.py

Transformer critic for planner-level offline RL.

It predicts:
    Q(s, task, action)


For Q, the tokenized inputs are:
    [CLS] + task tokens + robot tokens + object tokens + [ACTION]

Attention:
    - No causal mask is used.
    - Every valid token can attend to every other valid token.
    - A padding mask is used only to ignore padded task/object entries.

Robot state layout (37 values):
    0:7    base position + quaternion
    7:13   base linear velocity + angular velocity
    13:25  joint positions
    25:37  joint velocities

Object token layout is configurable through object_feature_dim. For the
current CSV, a natural 17-dimensional object vector is:
    position (3)
    quaternion (4)
    linear velocity (3)
    angular velocity (3)
    size (3)
    mass (1)

Action:
    skill ID
    target object ID (-1 means no object)
    goal_x
    goal_y
    goal_yaw
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
from transformers import AutoModel


@dataclass
class CriticConfig:
    # Language encoder
    text_model_name: str = "distilbert-base-uncased"
    freeze_text_encoder: bool = True

    # State Transformer
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1

    # Input dimensions
    robot_state_dim: int = 37
    object_feature_dim: int = 13

    # Structured action
    num_skills: int = 2
    max_object_id: int = 6

    # Used to normalize goal coordinates before action projection
    room_x_limit: float = 1.5
    room_y_limit: float = 2.5

    object_physical_dim: int = 224
    object_id_embedding_dim: int = 32

@dataclass
class CriticOutput:
    q_value: Tensor  # [batch]
    #v_value: Tensor  # [batch]


class ProjectionMLP(nn.Module):
    """Small MLP used to project one semantic quantity into one token."""

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
    Shared Transformer critic.

    V is computed from:
        task + robot + objects

    Q is computed from:
        task + robot + objects + current action

    The same Transformer parameters are used for both passes.
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
        # --------------------------------------------------------------
        # Task-language encoder
        # --------------------------------------------------------------
        self.text_encoder = AutoModel.from_pretrained(
            config.text_model_name
        )
        text_hidden_dim = int(self.text_encoder.config.hidden_size)

        self.task_projection = nn.Linear(
            text_hidden_dim,
            config.d_model,
        )

        if config.freeze_text_encoder:
            self.text_encoder.requires_grad_(False)

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

        self.object_physical_projection = ProjectionMLP(
            input_dim=config.object_feature_dim,
            output_dim=config.object_physical_dim,
        )

        #we use the same for the scene objects and the action id so the model understands the relationship
        self.object_id_embedding = nn.Embedding(
            num_embeddings=config.max_object_id + 1,
            embedding_dim=config.object_id_embedding_dim,
            padding_idx=0,
        )
        # --------------------------------------------------------------
        # Structured current-action token
        # --------------------------------------------------------------
        skill_dim = config.d_model // 4
        object_id_dim = config.object_id_embedding_dim
        goal_dim = config.d_model - skill_dim - object_id_dim

        self.skill_embedding = nn.Embedding(
            num_embeddings=config.num_skills,
            embedding_dim=skill_dim,
        )

        self.goal_projection = ProjectionMLP(
            input_dim=4,
            output_dim=goal_dim,
            hidden_dim=config.d_model,
            dropout=config.dropout,
        )

        #projected only once
        """self.action_projection = ProjectionMLP(
            input_dim=config.d_model,
            output_dim=config.d_model,
            dropout=config.dropout,
        )"""

        # --------------------------------------------------------------
        # Shared Transformer
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
        # Scalar heads
        # --------------------------------------------------------------
        """self.v_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 1),
        )"""

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

    def train(self, mode: bool = True) -> "StateActionTransformerCritic":
        """
        Keep a frozen language encoder in eval mode so its dropout is disabled.
        """
        super().train(mode)

        if self.config.freeze_text_encoder:
            self.text_encoder.eval()

        return self

    def encode_task(
        self,
        task_input_ids: Tensor,
        task_attention_mask: Tensor,
    ) -> Tensor:
        """
        Returns contextual language tokens of shape [B, L, d_model].
        """
        if self.config.freeze_text_encoder:
            with torch.no_grad():
                text_output = self.text_encoder(
                    input_ids=task_input_ids,
                    attention_mask=task_attention_mask,
                )
        else:
            text_output = self.text_encoder(
                input_ids=task_input_ids,
                attention_mask=task_attention_mask,
            )

        return self.task_projection(
            text_output.last_hidden_state
        )

    def encode_robot(self, robot_state: Tensor) -> Tensor:
        """
        Converts [B, 37] into four robot tokens [B, 4, d_model].
        """
        if robot_state.ndim != 2:
            raise ValueError(
                "robot_state must have shape [batch, 37]."
            )

        if robot_state.shape[-1] != 37:
            raise ValueError(
                f"Expected robot_state[..., 37], got "
                f"{tuple(robot_state.shape)}."
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
    ) -> Tensor:
        physical_part = self.object_physical_projection(
            objects
        )  # [B, N, 224]

        id_part = self.object_id_embedding(
            object_ids.long()
        )

        object_tokens = torch.cat(
            [physical_part, id_part],
            dim=-1,
        )  # [B, N, 256]

        return object_tokens

    def encode_action(
        self,
        action_skill: Tensor,
        action_object_id: Tensor,
        action_goal: Tensor,
        scene_object_ids: Tensor,
        object_mask: Tensor,
    ) -> Tensor:
        """
        Build one action token per batch item.

        Conventions
        -----------
        skill == 0:
            navigation
            action_object_id must be -1
            embedding ID becomes 0 ("no target object")

        skill == 1:
            manipulation
            action_object_id must be >= 1
            target ID must exist among the valid scene objects

        Inputs
        ------
        action_skill:
            [B]

        action_object_id:
            [B]

        action_goal:
            [B, 3] containing [goal_x, goal_y, goal_yaw]

        scene_object_ids:
            [B, N]
            Real objects use IDs >= 1.
            Padded slots use ID 0.

        object_mask:
            [B, N]
            True for real objects, False for padded slots.
        """
        if action_goal.ndim != 2 or action_goal.shape[-1] != 3:
            raise ValueError(
                "action_goal must have shape [batch, 3]."
            )

        if scene_object_ids.ndim != 2:
            raise ValueError(
                "scene_object_ids must have shape [batch, num_objects]."
            )

        if object_mask.shape != scene_object_ids.shape:
            raise ValueError(
                "object_mask must have the same shape as scene_object_ids."
            )

        action_skill = action_skill.long()
        action_object_id = action_object_id.long()
        scene_object_ids = scene_object_ids.long()
        object_mask = object_mask.bool()

        if action_skill.ndim != 1:
            raise ValueError(
                "action_skill must have shape [batch]."
            )

        if action_object_id.ndim != 1:
            raise ValueError(
                "action_object_id must have shape [batch]."
            )

        batch_size = action_skill.shape[0]

        if (
            action_object_id.shape[0] != batch_size
            or action_goal.shape[0] != batch_size
            or scene_object_ids.shape[0] != batch_size
        ):
            raise ValueError(
                "All action and scene inputs must have the same batch size."
            )

        # Assumed skill mapping.
        navigation_mask = action_skill == 0
        manipulation_mask = action_skill == 1

        if torch.any(~(navigation_mask | manipulation_mask)):
            raise ValueError(
                "action_skill must be 0 (navigation) or 1 (manipulation)."
            )

        # Navigation must not target an object.
        if torch.any(
            navigation_mask & (action_object_id != -1)
        ):
            bad_indices = torch.nonzero(
                navigation_mask & (action_object_id != -1),
                as_tuple=False,
            ).flatten().tolist()

            raise ValueError(
                "Navigation actions must have action_object_id == -1. "
                f"Invalid batch entries: {bad_indices}"
            )

        # Manipulation must target a real object ID.
        if torch.any(
            manipulation_mask & (action_object_id < 1)
        ):
            bad_indices = torch.nonzero(
                manipulation_mask & (action_object_id < 1),
                as_tuple=False,
            ).flatten().tolist()

            raise ValueError(
                "Manipulation actions must have action_object_id >= 1. "
                f"Invalid batch entries: {bad_indices}"
            )

        if torch.any(action_object_id > self.config.max_object_id):
            raise ValueError(
                "action_object_id exceeds max_object_id."
            )

        # Real scene-object slots must have IDs >= 1.
        if torch.any(
            object_mask & (scene_object_ids < 1)
        ):
            raise ValueError(
                "Valid scene objects must have IDs >= 1. "
                "ID 0 is reserved for padded/no-object entries."
            )

        # Padded slots should use ID 0.
        if torch.any(
            (~object_mask) & (scene_object_ids != 0)
        ):
            raise ValueError(
                "Padded object slots must have object ID 0."
            )

        # Check that every manipulation target exists in its scene.
        target_matches = (
            scene_object_ids
            == action_object_id.unsqueeze(1)
        ) & object_mask

        target_exists = target_matches.any(dim=1)

        if torch.any(
            manipulation_mask & ~target_exists
        ):
            bad_indices = torch.nonzero(
                manipulation_mask & ~target_exists,
                as_tuple=False,
            ).flatten().tolist()

            missing_targets = action_object_id[
                manipulation_mask & ~target_exists
            ].tolist()

            raise ValueError(
                "A manipulation action targets an object that is not "
                "present in the corresponding scene. "
                f"Batch entries: {bad_indices}; "
                f"missing target IDs: {missing_targets}"
            )

        # -1 is represented by embedding index 0.
        # Real object IDs retain their actual values.
        embedding_id = torch.where(
            navigation_mask,
            torch.zeros_like(action_object_id),
            action_object_id,
        )

        skill_token_part = self.skill_embedding(
            action_skill
        )

        object_token_part = self.object_id_embedding(
            embedding_id
        )

        goal_x = (
            action_goal[:, 0]
            / self.config.room_x_limit
        )
        goal_y = (
            action_goal[:, 1]
            / self.config.room_y_limit
        )
        goal_yaw = action_goal[:, 2]

        goal_features = torch.stack(
            [
                goal_x,
                goal_y,
                torch.sin(goal_yaw),
                torch.cos(goal_yaw),
            ],
            dim=-1,
        )

        goal_token_part = self.goal_projection(
            goal_features
        )

        action_features = torch.cat(
            [
                skill_token_part,
                object_token_part,
                goal_token_part,
            ],
            dim=-1,
        )

        return action_features
        """return self.action_projection(
            action_features
        )"""

    def build_tokens(
        self,
        task_tokens: Tensor,
        task_attention_mask: Tensor,
        robot_tokens: Tensor,
        object_tokens: Tensor,
        object_mask: Tensor,
        action_token: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Constructs the Transformer input and padding mask.

        There is no causal attention mask. All valid tokens can attend to all
        other valid tokens.

        The returned padding mask only prevents attention to padded task or
        object entries.
        """
        batch_size = task_tokens.shape[0]
        device = task_tokens.device

        if object_mask.shape != object_tokens.shape[:2]:
            raise ValueError(
                "object_mask must have shape [batch, num_objects]."
            )

        cls = self.cls_token.expand(batch_size, -1, -1)

        token_blocks = [
            cls,
            task_tokens,
            robot_tokens,
            object_tokens,
        ]

        type_id_blocks = [
            torch.full(
                (1,),
                self.CLS_TYPE,
                dtype=torch.long,
                device=device,
            ),
            torch.full(
                (task_tokens.shape[1],),
                self.TASK_TYPE,
                dtype=torch.long,
                device=device,
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
        ]

        padding_blocks = [
            torch.zeros(
                batch_size,
                1,
                dtype=torch.bool,
                device=device,
            ),
            ~task_attention_mask.bool(),
            torch.zeros(
                batch_size,
                robot_tokens.shape[1],
                dtype=torch.bool,
                device=device,
            ),
            ~object_mask.bool(),
        ]

        if action_token is not None:
            token_blocks.append(action_token.unsqueeze(1))

            type_id_blocks.append(
                torch.full(
                    (1,),
                    self.ACTION_TYPE,
                    dtype=torch.long,
                    device=device,
                )
            )

            padding_blocks.append(
                torch.zeros(
                    batch_size,
                    1,
                    dtype=torch.bool,
                    device=device,
                )
            )

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
        """
        Returns the final [CLS] representation.

        No `mask` argument is passed, so the attention is fully bidirectional.
        """
        encoded = self.transformer(
            src=tokens,
            src_key_padding_mask=padding_mask,
        )

        return encoded[:, 0]

    def forward(
        self,
        task_input_ids: Tensor,
        task_attention_mask: Tensor,
        robot_state: Tensor,
        objects: Tensor,
        object_ids: Tensor,
        object_mask: Tensor,
        action_skill: Tensor,
        action_object_id: Tensor,
        action_goal: Tensor,
    ) -> CriticOutput:

        task_tokens = self.encode_task(
            task_input_ids=task_input_ids,
            task_attention_mask=task_attention_mask,
        )

        robot_tokens = self.encode_robot(robot_state)

        object_tokens = self.encode_objects(
            objects=objects,
            object_ids=object_ids,
        )

        action_token = self.encode_action(
            action_skill=action_skill,
            action_object_id=action_object_id,
            action_goal=action_goal,
            scene_object_ids=object_ids,
            object_mask=object_mask,
        )

        q_tokens, q_padding_mask = self.build_tokens(
            task_tokens=task_tokens,
            task_attention_mask=task_attention_mask,
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

        return CriticOutput(
            q_value=q_value,
        )


if __name__ == "__main__":
    # Shape test.
    config = CriticConfig(
        object_feature_dim=13,
        freeze_text_encoder=True,
    )

    critic = StateActionTransformerCritic(config)

    batch_size = 3
    task_length = 16
    max_objects = 6

    dummy_batch = {
        "task_input_ids": torch.ones(
            batch_size,
            task_length,
            dtype=torch.long,
        ),
        "task_attention_mask": torch.ones(
            batch_size,
            task_length,
            dtype=torch.long,
        ),
        "robot_state": torch.randn(
            batch_size,
            37,
        ),
        "objects": torch.randn(
            batch_size,
            max_objects,
            config.object_feature_dim,
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
            [-1, 2, 4],
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

    output = critic(**dummy_batch)

    print("Q shape:", output.q_value.shape)
    print("V shape:", output.v_value.shape)
    print("Q values:", output.q_value)
    print("V values:", output.v_value)
