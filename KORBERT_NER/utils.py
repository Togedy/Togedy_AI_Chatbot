import os
import torch
import numpy as np
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight

def get_label_list(label_path="./KORBERT_NER/data/label.txt"):
    with open(label_path, encoding="utf-8") as f:
        labels = f.read().splitlines()
    return labels

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id is None:
            new_labels.append(-100)
        elif word_id != current_word:
            new_labels.append(labels[word_id])
            current_word = word_id
        else:
            label = labels[word_id]
            if label.startswith("B-"):
                label = label.replace("B-", "I-")
            new_labels.append(label)
    return new_labels

def compute_metrics(pred):
    from seqeval.metrics import f1_score, accuracy_score, classification_report

    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    true_labels = []
    true_preds = []

    for lab, pred in zip(labels, preds):
        true_lab = []
        true_pred = []
        for l, p in zip(lab, pred):
            if l != -100:
                true_lab.append(l)
                true_pred.append(p)
        true_labels.append(true_lab)
        true_preds.append(true_pred)

    return {
        "accuracy": accuracy_score(true_labels, true_preds),
        "f1": f1_score(true_labels, true_preds),
        "report": classification_report(true_labels, true_preds, output_dict=False)
    }

def compute_class_weights(dataset, num_labels):
    all_labels = []
    for item in dataset:
        labels = item['labels']
        all_labels.extend([label for label in labels if label != -100])

    label_counts = Counter(all_labels)
    total = sum(label_counts.values())
    weights = [0.0] * num_labels
    for i in range(num_labels):
        weights[i] = total / (num_labels * label_counts.get(i, 1))

    return torch.tensor(weights, dtype=torch.float)

def read_data(mode):
    path = f"./KORBERT_NER/data/{mode}.tsv"
    sentences = []
    tags = []

    with open(path, "r", encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            line = line.strip()
            if not line:
                if words:
                    sentences.append(words)
                    tags.append(labels)
                    words = []
                    labels = []
            else:
                splits = line.split("\t")
                if len(splits) == 2:
                    words.append(splits[0])
                    labels.append(splits[1])
        if words:
            sentences.append(words)
            tags.append(labels)

    return sentences, tags
