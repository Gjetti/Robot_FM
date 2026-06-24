# datasets/loader.py

import json
from sklearn.model_selection import train_test_split

def load_dataset(dataset_cfg):

    dataset_path = dataset_cfg["path"]
    dataset_type = dataset_cfg["type"]

    if dataset_type in ["alpaca", "chatml"]:

        data = []

        with open(dataset_path, "r") as f:
            for line in f:
                data.append(json.loads(line))

        return data

    raise ValueError(
        f"Unsupported dataset type: {dataset_type}"
    )

def create_train_eval_split(
    data,
    eval_ratio=0.05,
    seed=42,
):

    train_data, eval_data = train_test_split(
        data,
        test_size=eval_ratio,
        random_state=seed,
        shuffle=True,
    )

    return train_data, eval_data