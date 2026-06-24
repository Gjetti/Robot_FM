""" For now this only has LoRA, we might want to add QLoRA later."""
from peft import LoraConfig, get_peft_model


def add_lora(
    model,
    lora_cfg
):

    config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        task_type="CAUSAL_LM",

        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, config)

    return model