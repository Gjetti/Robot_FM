from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

RUN_NAME = "Qwen2.5-1.5B-Instruct_lr2e-05_r24_20260623_191554"
# training run to merge
LORA_PATH = f"checkpoints/{RUN_NAME}/best_adapter"


OUTPUT_PATH = f"checkpoints/merged/{RUN_NAME}"

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype="auto",
    device_map="cpu",
)

model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH,
)

merged_model = model.merge_and_unload()

import os

os.makedirs(OUTPUT_PATH, exist_ok=True)

merged_model.save_pretrained(OUTPUT_PATH)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.save_pretrained(OUTPUT_PATH)

print(f"Merged model saved to {OUTPUT_PATH}")