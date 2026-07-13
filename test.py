"""
this code will be used as a testing base for the models. Since the first work was evaluated on collected_demos_for_physics_rollouts_filtered.csv.
we will follow the same convention here. This code infers the plan from the "same" trajectories we used to do the eval in the corl paper and outputs 2 files:
1- a csv containing the plan, traj_id, and the latency.
2- a csv containing the initial env state with the plan to easily check the metrics.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer




# ==========================================================
# CONFIG
# ==========================================================

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

#change name if needed
MODEL_PATH = (
    "checkpoints/merged/"
    "Qwen2.5-1.5B-Instruct_lr2e-05_r512_20260710_092126"
)

INPUT_CSV = "/home/quads/data llms/collected_demos_for_physics_rollouts_filtered.csv"

MAX_NEW_TOKENS = 150
#NOTE that it is best to have do_sample=False to check the models plans deterministically. however it is also beneficial to check the stochastic planning of the model!
DO_SAMPLE = False
#these are useless if we use DO_SAMPLE = False, since ti will use greedy decoding.
TEMPERATURE = 0.8
TOP_P = 0.95

NUM_PER_BUCKET = 25
BUCKET_SIZE = 1000
MAX_TRAJ_ID = 8000

SEED = 1

REFERENCE_FRAME = "world"   # "world" or "robot" in the dataset
# ==========================================================
# Geometry helpers for the transformation
# ==========================================================

import math
import re

ROBOT_POSE_RE = re.compile(
    r"\(x=([-0-9.]+),y=([-0-9.]+),yaw=([-0-9.]+)\)"
)

OBJECT_POSE_RE = re.compile(
    r"\(x=([-0-9.]+),y=([-0-9.]+),yaw=([-0-9.]+)\)"
)


def wrap_angle(angle_deg):
    while angle_deg > 180:
        angle_deg -= 360
    while angle_deg <= -180:
        angle_deg += 360
    return angle_deg


def world_to_robot(
    x_world,
    y_world,
    yaw_world,
    robot_x,
    robot_y,
    robot_yaw,
):
    dx = x_world - robot_x
    dy = y_world - robot_y

    theta = math.radians(robot_yaw)

    x_robot = (
        math.cos(theta) * dx
        + math.sin(theta) * dy
    )

    y_robot = (
        math.sin(theta) * dx
        - math.cos(theta) * dy
    )

    yaw_robot = wrap_angle(
        yaw_world - robot_yaw
    )

    return (
        round(x_robot, 3),
        round(y_robot, 3),
        round(yaw_robot, 1),
    )

def parse_robot_pose(robot_string):

    m = ROBOT_POSE_RE.search(robot_string)

    if m is None:
        raise RuntimeError(
            f"Could not parse robot pose:\n{robot_string}"
        )

    return (
        float(m.group(1)),
        float(m.group(2)),
        float(m.group(3)),
    )

def robot_to_world(
    x_robot,
    y_robot,
    yaw_robot,
    robot_x,
    robot_y,
    robot_yaw,
):
    theta = math.radians(robot_yaw)

    dx = (
        math.cos(theta) * x_robot
        + math.sin(theta) * y_robot
    )

    dy = (
        math.sin(theta) * x_robot
        - math.cos(theta) * y_robot
    )

    x_world = robot_x + dx
    y_world = robot_y + dy

    yaw_world = wrap_angle(
        yaw_robot + robot_yaw
    )

    return (
        round(x_world, 3),
        round(y_world, 3),
        round(yaw_world, 1),
    )

POSE_RE = re.compile(
    r"\(x=([-0-9.]+),y=([-0-9.]+),yaw=([-0-9.]+)\)"
)

def robot_plan_to_world(
    plan,
    robot_x,
    robot_y,
    robot_yaw,
):
    def repl(match):

        x = float(match.group(1))
        y = float(match.group(2))
        yaw = float(match.group(3))

        xw, yw, yaww = robot_to_world(
            x,
            y,
            yaw,
            robot_x,
            robot_y,
            robot_yaw,
        )

        return (
            f"(x={xw},y={yw},yaw={yaww})"
        )

    return POSE_RE.sub(repl, plan)


# ==========================================================
# OUTPUT FOLDER
# ==========================================================

run_name = Path(MODEL_PATH).name

output_dir = Path("test_results") / run_name
output_dir.mkdir(parents=True, exist_ok=True)

predictions_csv = output_dir / "predictions.csv"
rollout_csv = output_dir / "rollout_dataset.csv"


# ==========================================================
# REPRODUCIBILITY
# ==========================================================

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv(INPUT_CSV)

df0 = df[df["state_index"] == 0].copy()

df0 = df0[df0["trajectory_id"] < MAX_TRAJ_ID]

df0 = df0.sort_values("trajectory_id")

df0["bucket"] = df0["trajectory_id"] // BUCKET_SIZE

sampled_list = []

for _, group in df0.groupby("bucket"):

    if len(group) == 0:
        continue

    n = min(NUM_PER_BUCKET, len(group))

    sampled_group = group.sample(
        n=n,
        random_state=SEED,
    )

    sampled_list.append(sampled_group)

sampled = pd.concat(sampled_list)

sampled = sampled.sort_values(
    "trajectory_id"
).reset_index(drop=True)

print("\n=== SELECTED TRAJECTORY IDS ===")
print(sampled["trajectory_id"].tolist())
print(f"Total selected: {len(sampled)}")


# ==========================================================
# LOAD MODEL
# ==========================================================

torch.cuda.empty_cache()

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    use_fast=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="cuda",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)

model.eval()


eos_ids = [
    tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

"""
print("eos_token:", tokenizer.eos_token)
print("eos_token_id:", tokenizer.eos_token_id)

print(
    "<|end_of_text|> ->",
    tokenizer.convert_tokens_to_ids("<|end_of_text|>")
)

print(
    "<|eot_id|> ->",
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
)

ids = tokenizer.encode(
    "[{skill:navigate_to}]<|end_of_text|>",
    add_special_tokens=False,
)

print(ids)
print(tokenizer.decode(ids))
"""
# ==========================================================
# HELPERS
# ==========================================================

def build_instruction(row):

    task = str(row["task"]).strip()

    if (
        task == ""
        or task.lower() == "nan"
    ):
        task = (
            "navigate to "
            "(x=0.0,y=0.0,yaw=0.0)"
        )

    if REFERENCE_FRAME == "world":
        return task

    robot_x, robot_y, robot_yaw = (
        parse_robot_pose(row["robot"])
    )

    m = ROBOT_POSE_RE.search(task)

    if m is None:
        return task

    gx = float(m.group(1))
    gy = float(m.group(2))
    gyaw = float(m.group(3))

    gx_r, gy_r, gyaw_r = world_to_robot(
        gx,
        gy,
        gyaw,
        robot_x,
        robot_y,
        robot_yaw,
    )

    return (
        f"navigate to "
        f"(x={gx_r},y={gy_r},yaw={gyaw_r})"
    )


"""def build_input_text(row):

    text = f"Robot state: {row['robot']}\n"
    text += "Objects:\n"

    for i in range(1, 21):

        col = f"object_{i}"

        if col in row and pd.notna(row[col]):
            text += f"{col}: {row[col]}\n"

    text += "Previous skill: NA\n"
    text += "Done: False\n"

    return text"""

def transform_object_string(
    object_string,
    robot_x,
    robot_y,
    robot_yaw,
):

    parts = object_string.rsplit(",", 3)

    if len(parts) != 4:
        return object_string

    pose = (
        parts[-3]
        + ","
        + parts[-2]
        + ","
        + parts[-1]
    )

    m = ROBOT_POSE_RE.search(pose)

    if m is None:
        return object_string

    ox = float(m.group(1))
    oy = float(m.group(2))
    oyaw = float(m.group(3))

    ox_r, oy_r, oyaw_r = world_to_robot(
        ox,
        oy,
        oyaw,
        robot_x,
        robot_y,
        robot_yaw,
    )

    prefix = object_string[
        : m.start()
    ]

    return (
        prefix
        + f"(x={ox_r},y={oy_r},yaw={oyaw_r})"
    )

def build_input_text(row):

    text = ""

    if REFERENCE_FRAME == "world":
        text += f"Robot state: {row['robot']}\n"

    text += "Objects:\n"

    if REFERENCE_FRAME == "robot":

        robot_x, robot_y, robot_yaw = (
            parse_robot_pose(row["robot"])
        )

    for i in range(1, 21):

        col = f"object_{i}"

        if col not in row:
            continue

        if pd.isna(row[col]):
            continue

        obj = str(row[col])

        if REFERENCE_FRAME == "robot":

            obj = transform_object_string(
                obj,
                robot_x,
                robot_y,
                robot_yaw,
            )

        text += f"{col}: {obj}\n"

    text += "Previous skill: NA\n"
    text += "Done: False\n"

    return text

def build_prompt(
    instruction,
    input_text,
    tokenizer=None,
    format_type="alpaca",
):

    if format_type == "alpaca":

        return f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""

    elif format_type == "chatml":

        messages = [
            {
                "role": "user",
                "content": f"{instruction}\n\n{input_text}",
            }
        ]

        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    else:
        raise ValueError(
            f"Unknown format: {format_type}"
        )

# ==========================================================
# INFERENCE
# ==========================================================

results = []
latencies = []

for idx, row in sampled.iterrows():

    traj_id = row["trajectory_id"]

    instruction = build_instruction(row)
    input_text = build_input_text(row)

    prompt = build_prompt(
        instruction,
        input_text,
        tokenizer=tokenizer,
        format_type="chatml",
    )
    #print(prompt)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    # warmup
    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=16,
        )

    torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            #temperature=TEMPERATURE,
            #top_p=TOP_P,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    torch.cuda.synchronize()

    latency = time.time() - start

    latencies.append(latency)

    """decoded = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )


    if "### Response:" in decoded:
        response = decoded.split(
            "### Response:"
        )[-1].strip()
    else:
        response = decoded.strip()
""" 
    # only keep generated tokens
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=False,
    ).strip()
    #print(response)
    EOS_MARKERS = [
        "<|end_of_text|>",
        "<|im_end|>",
    ]

    for marker in EOS_MARKERS:
        if marker in response:
            response = response.split(marker)[0].strip()

    #print(response)  

    results.append(
        {
            "trajectory_id": traj_id,
            "latency_sec": latency,
            "plan": response,
        }
    )

    print(
        f"[{idx + 1}/{len(sampled)}] "
        f"traj={traj_id} | "
        f"{latency:.3f}s"
    )


# ==========================================================
# SAVE PREDICTIONS
# ==========================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    predictions_csv,
    index=False,
)

print(
    f"\nSaved predictions to:\n{predictions_csv}"
)


# ==========================================================
# CREATE ROLLOUT DATASET(S)
# ==========================================================

rollout_df = sampled.copy()

plan_map = results_df.set_index(
    "trajectory_id"
)["plan"]

rollout_df["actions"] = rollout_df[
    "trajectory_id"
].map(plan_map)

columns = [
    "task",
    "trajectory_id",
    "state_index",
    "robot",
]

for i in range(1, 21):

    col = f"object_{i}"

    if col in rollout_df.columns:
        columns.append(col)

columns.append("actions")

rollout_df = rollout_df[columns]


# ----------------------------------------------------------
# WORLD FRAME DATASET (default behavior)
# ----------------------------------------------------------

if REFERENCE_FRAME == "world":

    rollout_df.to_csv(
        rollout_csv,
        index=False,
    )

    print(
        f"Saved rollout dataset to:\n{rollout_csv}"
    )

# ----------------------------------------------------------
# ROBOT FRAME DATASET + WORLD FRAME DATASET
# ----------------------------------------------------------

else:

    # save raw robot-frame outputs
    rollout_csv_robot = (
        output_dir / "rollout_dataset_robot.csv"
    )

    rollout_df.to_csv(
        rollout_csv_robot,
        index=False,
    )

    print(
        f"Saved robot-frame rollout dataset to:\n"
        f"{rollout_csv_robot}"
    )

    # create world-frame copy for visualization
    rollout_df_world = rollout_df.copy()

    world_actions = []

    for _, row in rollout_df_world.iterrows():

        robot_x, robot_y, robot_yaw = (
            parse_robot_pose(row["robot"])
        )

        world_actions.append(
            robot_plan_to_world(
                row["actions"],
                robot_x,
                robot_y,
                robot_yaw,
            )
        )

    rollout_df_world["actions"] = world_actions

    rollout_df_world.to_csv(
        rollout_csv,
        index=False,
    )

    print(
        f"Saved world-frame rollout dataset to:\n"
        f"{rollout_csv}"
    )

# ==========================================================
# LATENCY STATS
# ==========================================================

latencies = np.array(latencies)

print("\n=== LATENCY STATS ===")
print(f"Mean: {latencies.mean():.4f} s")
print(f"Std:  {latencies.std():.4f} s")
print(
    f"Mean ± Std: "
    f"{latencies.mean():.4f} ± {latencies.std():.4f} s"
)