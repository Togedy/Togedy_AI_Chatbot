import csv
from datasets import DatasetDict, Dataset

def read_tsv_file(file_path):
    sentences = []
    labels = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        tokens = []
        tags = []
        for line in reader:
            if len(line) == 0 or line[0].startswith("-DOCSTART-"):
                continue
            if len(line) == 1:
                if tokens:
                    sentences.append(tokens)
                    labels.append(tags)
                    tokens = []
                    tags = []
            else:
                tokens.append(line[0])
                tags.append(line[1])
        if tokens:
            sentences.append(tokens)
            labels.append(tags)
    return {"tokens": sentences, "labels": labels}

def load_dataset(train_path, test_path):
    train_data = read_tsv_file(train_path)
    test_data = read_tsv_file(test_path)
    dataset = DatasetDict({
        "train": Dataset.from_dict(train_data),
        "test": Dataset.from_dict(test_data)
    })
    return dataset
