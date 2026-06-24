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
    "Qwen2.5-1.5B-Instruct_lr2e-05_r24_20260623_191554"
)

INPUT_CSV = "/home/georges/Go2SDK-main/example/go2/low_level/collected_demos_for_physics_rollouts_filtered.csv"

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

    if task == "" or task.lower() == "nan":
        task = "navigate to (x=0.0,y=0.0,yaw=0.0)"

    extra = (
        " without colliding with any object. "
        "You must reason if a collision-free path exists. "
        "You may manipulate (PLANAR PUSHING) movable objects if necessary, "
        "but you must approach objects before manipulating them if they are far."
    )

    return task #+ extra


def build_input_text(row):

    text = f"Robot state: {row['robot']}\n"
    text += "Objects:\n"

    for i in range(1, 21):

        col = f"object_{i}"

        if col in row and pd.notna(row[col]):
            text += f"{col}: {row[col]}\n"

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
# CREATE ROLLOUT DATASET
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

rollout_df.to_csv(
    rollout_csv,
    index=False,
)

print(
    f"Saved rollout dataset to:\n{rollout_csv}"
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