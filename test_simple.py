from models.planner import load_model_and_tokenizer
import time
model, tokenizer = load_model_and_tokenizer(
    "Qwen/Qwen2.5-1.5B-Instruct"
)


messages = [
    {"role": "user", "content": "How can i compute the distance between 2 points?"}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(text, return_tensors="pt").to("cuda")


start = time.time()
outputs = model.llm.generate(
    **inputs,
    max_new_tokens=100,

    do_sample=False,
    eos_token_id=tokenizer.eos_token_id,
)
end = time.time()

elapsed = end - start
num_generated = outputs.shape[-1] - inputs["input_ids"].shape[-1]

print("\n=== PERFORMANCE ===")
print(f"Inference time: {elapsed:.3f} s")
print(f"Generated tokens: {num_generated}")
print(f"Tokens/sec: {num_generated / elapsed:.2f}")

print("\n=== MODEL OUTPUT ===\n")
generated = outputs[0][inputs.input_ids.shape[1]:]
print(tokenizer.decode(generated, skip_special_tokens=True))
#print(tokenizer.decode(outputs[0]))