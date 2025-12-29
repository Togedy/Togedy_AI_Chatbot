import os
import json
import random
import numpy as np
import torch

# TYPE 전용 라벨 파일 (B-TYPE, I-TYPE, O) — 이 경로/순서를 사용합니다.
LABEL_PATH = "./KORBERT_NER_TYPE/data/label.txt"

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_label_list():
    with open(LABEL_PATH, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    id2label = {i: lab for i, lab in enumerate(labels)}
    label2id = {lab: i for i, lab in enumerate(labels)}
    return labels, label2id, id2label

def compute_class_weights(examples, label2id):
    """
    tokenized Dataset의 labels에는 -100(무시 토큰)이 섞여 있음 → 제외하고 집계
    """
    num_labels = len(label2id)
    counts = np.zeros(num_labels, dtype=np.float64)
    labels_list = examples["labels"] if isinstance(examples["labels"], list) else list(examples["labels"])
    for sent in labels_list:
        for y in sent:
            if 0 <= y < num_labels:   # -100 무시
                counts[y] += 1
    counts = np.where(counts == 0, 1.0, counts)  # 분모 0 방지
    weights = counts.sum() / (counts * num_labels)
    return torch.tensor(weights, dtype=torch.float32)

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
