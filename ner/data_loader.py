import csv
from datasets import DatasetDict, Dataset
from sklearn.model_selection import train_test_split

def read_tsv_file(file_path):
    sentences = []
    labels = []
    tokens = []
    tags = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                if tokens:
                    sentences.append(tokens)
                    labels.append(tags)
                    tokens = []
                    tags = []
                continue

            if "\t" in line:
                splits = line.split("\t")
                if len(splits) == 2:
                    token, tag = splits
                    tokens.append(token)
                    tags.append(tag)

        # 마지막 문장 추가
        if tokens:
            sentences.append(tokens)
            labels.append(tags)

    return {"tokens": sentences, "labels": labels}


def load_dataset(train_path, test_path, valid_ratio=0.1):
    full_train_data = read_tsv_file(train_path)
    test_data = read_tsv_file(test_path)

    # Train/Validation split
    train_tokens, val_tokens, train_labels, val_labels = train_test_split(
        full_train_data["tokens"],
        full_train_data["labels"],
        test_size=valid_ratio,
        random_state=42
    )

    dataset = DatasetDict({
        "train": Dataset.from_dict({"tokens": train_tokens, "labels": train_labels}),
        "validation": Dataset.from_dict({"tokens": val_tokens, "labels": val_labels}),
        "test": Dataset.from_dict(test_data)
    })

    return dataset
