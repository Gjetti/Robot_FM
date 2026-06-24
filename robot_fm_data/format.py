# datasets/formatters.py

class AlpacaFormatter:
    """Alpaca style data format (instruction, input, output/response)"""
    def format(self, sample):

        instruction = sample["instruction"]
        input_text = sample["input"]
        output_text = sample["output"]

        prompt = (
            f"Instruction:\n{instruction}\n\n"
            f"Input:\n{input_text}\n\n"
            f"Response:\n"
        )

        return prompt, output_text
    

""" we could add different data formats later to test their effect on the model."""

class ChatMLFormatter:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def format(self, sample):

        messages = sample["messages"]

        prompt = self.tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
        )

        response = messages[-1]["content"]
        if not response.endswith(self.tokenizer.eos_token):
            response += self.tokenizer.eos_token
        return prompt, response