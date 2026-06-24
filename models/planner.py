"""This code is used to import the base model, i.e., the LLM backbone."""

import torch
import torch.nn as nn

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


class PlannerModel(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()

        self.model_name = model_name

        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
        )

    def forward(self, **kwargs):
        return self.llm(**kwargs)

    @property
    def hidden_size(self):
        return self.llm.config.hidden_size
    
def load_model_and_tokenizer(model_name):

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = PlannerModel(model_name)

    return model, tokenizer