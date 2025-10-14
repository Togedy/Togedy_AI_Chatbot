# KORBERT_NER/utils.py
import torch
from typing import List, Tuple, Dict
from collections import Counter
import random
import numpy as np

def get_label_list(label_path: str):
    labels = [l.strip() for l in open(label_path, "r", encoding="utf-8").read().splitlines() if l.strip()]
    label_to_id = {l:i for i,l in enumerate(labels)}
    id_to_label = {i:l for l,i in label_to_id.items()}
    return labels, label_to_id, id_to_label

def compute_class_weights(label_list: List[str], train_ds) -> torch.Tensor:
    """
    train_ds: tokenized train dataset (contains 'labels')
    """
    counter = Counter()
    total = 0
    for ex in train_ds:
        for y in ex["labels"]:
            if y != -100:
                counter[y] += 1
                total += 1
    weights = []
    for i, lab in enumerate(label_list):
        c = counter.get(i, 0)
        # 빈도 역수 안정화
        w = 1.0 / max(1, c)
        weights.append(w)
    # 정규화
    t = torch.tensor(weights, dtype=torch.float)
    t = t * (len(t) / t.sum())
    return t

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
