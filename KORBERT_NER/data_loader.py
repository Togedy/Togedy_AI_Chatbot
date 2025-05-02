import torch
from torch.utils.data import Dataset
from utils import get_label_list

class NERDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def load_and_cache_examples(tokenizer, label_list, mode="train"):
    path = f"./KORBERT_NER/data/{mode}.tsv"
    sentences, tags = [], []
    with open(path, encoding="utf-8") as f:
        sentence, tag = [], []
        for line in f:
            if line.strip() == "":
                if sentence:
                    sentences.append(sentence)
                    tags.append(tag)
                    sentence, tag = [], []
            else:
                token, label = line.strip().split("\t")
                sentence.append(token)
                tag.append(label)
        if sentence:
            sentences.append(sentence)
            tags.append(tag)

    label_to_id = {label: i for i, label in enumerate(label_list)}

    encoded_inputs = tokenizer(
        sentences,
        is_split_into_words=True,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    labels = []
    for i, label in enumerate(tags):
        word_ids = encoded_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id.get(label[word_idx], 0))
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    return NERDataset(encoded_inputs, labels)
