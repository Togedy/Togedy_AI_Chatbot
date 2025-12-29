# -*- coding: utf-8 -*-
import torch
from collections import Counter

LABEL_PATH = "./KORBERT_NER_KEYWORD/data/label.txt"

def load_label_list(label_path: str = LABEL_PATH):
    with open(label_path, "r", encoding="utf-8") as f:
        labels = [ln.strip() for ln in f if ln.strip()]
    # 예: ["O","B-KEYWORD","I-KEYWORD"]
    id2label = {i: lab for i, lab in enumerate(labels)}
    label2id = {lab: i for i, lab in enumerate(labels)}
    return labels, label2id, id2label

def compute_class_weights(all_label_ids, num_labels, ignore_index=-100):
    """
    라벨 불균형 완화용 가중치. pad/ignored는 제외.
    빈도 역수(√) 방식으로 과도한 편향을 누름.
    """
    flat = [lid for seq in all_label_ids for lid in seq if lid != ignore_index]
    cnt = Counter(flat)
    # 최소 보호
    counts = torch.tensor([max(cnt.get(i, 1), 1) for i in range(num_labels)], dtype=torch.float)
    inv = 1.0 / torch.sqrt(counts)
    weights = inv / inv.sum() * num_labels
    return weights
