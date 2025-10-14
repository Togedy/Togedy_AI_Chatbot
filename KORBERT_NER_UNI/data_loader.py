# KORBERT_NER/data_loader.py
from typing import List, Tuple, Dict
from datasets import Dataset, DatasetDict

def _read_tsv(path: str) -> List[Dict]:
    """
    TSV: token<TAB>label, 문장 사이 빈 줄
    """
    samples = []
    tokens, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    samples.append({"tokens": tokens, "labels": labels})
                tokens, labels = [], []
                continue
            tok, lab = line.split("\t")
            tokens.append(tok)
            labels.append(lab)
    if tokens:
        samples.append({"tokens": tokens, "labels": labels})
    return samples

def load_dataset(train_path: str, test_path: str, valid_ratio: float = 0.1) -> DatasetDict:
    train_samples = _read_tsv(train_path)
    test_samples  = _read_tsv(test_path)

    full_train = Dataset.from_list(train_samples)
    # 간단 분할
    n = len(full_train)
    n_valid = max(1, int(n * valid_ratio))
    valid = full_train.select(range(n_valid))
    train = full_train.select(range(n_valid, n))

    test = Dataset.from_list(test_samples)
    return DatasetDict({"train": train, "validation": valid, "test": test})
