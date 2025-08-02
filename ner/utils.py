from collections import Counter
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

def load_label(path):
    label_list = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            label_list.append(line.strip())
    return label_list

def compute_class_weights(label_list, dataset):
    all_labels = []
    for example in dataset:
        all_labels.extend([l for l in example["labels"] if l != -100])

    label_counts = Counter(all_labels)
    total = sum(label_counts.values())

    class_weights = [0.0] * len(label_list)
    for idx in range(len(label_list)):
        class_weights[idx] = total / (len(label_list) * (label_counts.get(idx, 1)))
    return torch.tensor(class_weights, dtype=torch.float)

def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(axis=-1)

    true_labels = []
    true_predictions = []

    for pred, label in zip(predictions, labels):
        for p_i, l_i in zip(pred, label):
            if l_i != -100:
                true_labels.append(l_i)
                true_predictions.append(p_i)

    return {
        "precision": precision_score(true_labels, true_predictions, average="macro", zero_division=0),
        "recall": recall_score(true_labels, true_predictions, average="macro", zero_division=0),
        "f1": f1_score(true_labels, true_predictions, average="macro", zero_division=0),
    }

def get_label_list(label_path="./data/label.txt"):
    with open(label_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    label_to_id = {label: i for i, label in enumerate(labels)}
    id_to_label = {i: label for label, i in label_to_id.items()}
    return labels, label_to_id, id_to_label
