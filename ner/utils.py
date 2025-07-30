from collections import Counter
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

def load_label(path):
    label_list = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            label_list.append(line.strip())
    return label_list

def compute_class_weights(label_list, datasets):
    # datasets["train"] is a list of label strings (extracted from tokenized dataset)
    label_counts = Counter(datasets["train"])
    total = sum(label_counts.values())

    class_weights = [0.0] * len(label_list)
    for idx, label in enumerate(label_list):
        # Add 1 to denominator to avoid division by zero
        class_weights[idx] = total / (len(label_list) * (label_counts.get(label, 1)))
    return class_weights

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
