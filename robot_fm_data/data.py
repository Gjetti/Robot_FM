# datasets/data.py

from torch.utils.data import Dataset

class PlannerDataset(Dataset):

    def __init__(
        self,
        data,
        tokenizer,
        formatter,
        max_length,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.formatter = formatter
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        sample = self.data[idx]

        prompt, response = self.formatter.format(sample)

        full_text = prompt + response

        full_tokens = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        prompt_tokens = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = full_tokens["input_ids"].squeeze(0)
        attention_mask = full_tokens["attention_mask"].squeeze(0)

        labels = input_ids.clone()

        prompt_length = prompt_tokens["input_ids"].shape[1]

        # ignore prompt
        labels[:prompt_length] = -100

        # ignore padding
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    
