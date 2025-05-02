import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight

def get_label_list(label_path):
    with open(label_path, encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

def compute_class_weights(dataset, num_labels):
    all_labels = []
    for item in dataset:
        labels = item["labels"].numpy()
        all_labels.extend([label for label in labels if label != -100])
    all_labels = np.array(all_labels)
    unique_labels = np.unique(all_labels)
    weights = compute_class_weight(class_weight='balanced', classes=unique_labels, y=all_labels)
    full_weights = np.ones(num_labels)
    for i, cls in enumerate(unique_labels):
        full_weights[cls] = weights[i]
    return torch.tensor(full_weights)

def compute_metrics(p):
    predictions, labels = p
    preds = np.argmax(predictions, axis=2)

    true_labels = []
    true_preds = []

    for pred, label in zip(preds, labels):
        for p_, l_ in zip(pred, label):
            if l_ != -100:
                true_preds.append(p_)
                true_labels.append(l_)

    accuracy = np.mean(np.array(true_preds) == np.array(true_labels))
    return {"accuracy": accuracy}
