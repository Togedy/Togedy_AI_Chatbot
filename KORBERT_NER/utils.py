import torch
from sklearn.metrics import precision_recall_fscore_support

def encode_examples(sentences, labels, tokenizer, label_list):
    if isinstance(sentences, tuple):
        sentences = [list(s) for s in sentences]
    if isinstance(labels, tuple):
        labels = [list(l) for l in labels]

    assert all(isinstance(s, list) for s in sentences)
    assert all(isinstance(w, str) for s in sentences for w in s if w is not None)

    encodings = tokenizer(
        sentences,
        is_split_into_words=True,
        return_offsets_mapping=True,
        padding=True,
        truncation=True,
        max_length=512
    )

    encoded_labels = []
    label2id = {label: idx for idx, label in enumerate(label_list)}

    for i, label in enumerate(labels):
        word_ids = encodings.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                tag = label[word_idx]
                if tag not in label2id:
                    raise ValueError(f"Label '{tag}' not found in label_list.")
                label_ids.append(label2id[tag])
            else:
                tag = label[word_idx]
                if tag.startswith("I-") and tag in label2id:
                    label_ids.append(label2id[tag])
                else:
                    label_ids.append(-100)
            previous_word_idx = word_idx

        encoded_labels.append(label_ids)

    encodings = {
        key: torch.tensor(val) for key, val in encodings.items()
        if key not in ["offset_mapping", "token_type_ids"]
    }
    encoded_labels = torch.tensor(encoded_labels)

    return encodings, encoded_labels

def get_label_list(label_path):
    with open(label_path, encoding="utf-8") as f:
        labels = f.read().splitlines()
    return labels

def compute_metrics(p, label_list):
    predictions, labels = p
    predictions = predictions.argmax(axis=-1)

    true_labels = []
    true_preds = []

    for pred, label in zip(predictions, labels):
        for p_i, l_i in zip(pred, label):
            if l_i != -100:
                true_labels.append(label_list[l_i])
                true_preds.append(label_list[p_i])

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, true_preds, average="micro"
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
