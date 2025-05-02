import os
import torch
from torch.utils.data import Dataset
from utils import get_label_list, read_data

def encode_data(sentences, tags, tokenizer, label_to_id):
    encoded_inputs = tokenizer(sentences, is_split_into_words=True, padding=True, truncation=True, return_tensors="pt")
    encoded_labels = []

    for i, label in enumerate(tags):
        word_ids = encoded_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id[label[word_idx]])
            else:
                label_ids.append(label_to_id[label[word_idx]] if label[word_idx].startswith("B-") else label_to_id[label[word_idx]])
            previous_word_idx = word_idx
        encoded_labels.append(label_ids)

    encoded_inputs["labels"] = torch.tensor(encoded_labels)
    return encoded_inputs

class NERDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

def load_and_cache_examples(tokenizer, label_list, mode="train"):
    label_to_id = {label: i for i, label in enumerate(label_list)}
    sentences, tags = read_data(mode)
    encoded_inputs = encode_data(sentences, tags, tokenizer, label_to_id)
    return NERDataset(encoded_inputs)
