import torch
from torch.utils.data import Dataset
from utils import read_ner_data
from transformers import PreTrainedTokenizer


class NERDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def encode_data(sentences, tags, tokenizer: PreTrainedTokenizer, label_to_id):
    encodings = tokenizer(sentences, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True)
    encoded_labels = []

    for i, label in enumerate(tags):
        word_ids = encodings.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id.get(label[word_idx], 0))
            else:
                label_ids.append(label_to_id.get(label[word_idx], 0))
            previous_word_idx = word_idx

        encoded_labels.append(label_ids)

    return encodings, encoded_labels


def load_and_cache_examples(tokenizer, label_list, mode="train"):
    file_path = f"./KORBERT_NER/data/{mode}.tsv"
    sentences, tags = read_ner_data(file_path)
    label_to_id = {label: i for i, label in enumerate(label_list)}
    encodings, encoded_labels = encode_data(sentences, tags, tokenizer, label_to_id)
    dataset = NERDataset(encodings, encoded_labels)
    return dataset
