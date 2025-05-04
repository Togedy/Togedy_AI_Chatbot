import os
import torch
from torch.utils.data import Dataset
from utils import get_label_list, read_data

def encode_data(sentences, tags, tokenizer, label_to_id, max_length=128):
    input_ids = []
    attention_masks = []
    label_ids = []

    for words, labels in zip(sentences, tags):
        # 각 문장을 토큰 단위로 분해
        tokens = []
        label_for_tokens = []

        for word, label in zip(words, labels):
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            # 첫 토큰은 원래 라벨, 나머지는 I-로
            label_for_tokens.extend(
                [label_to_id[label]] + 
                [label_to_id[label.replace("B-", "I-")] if label.startswith("B-") else label_to_id[label]] * (len(word_tokens) - 1)
            )

        # special tokens 추가
        tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
        input_id = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_id)

        # 라벨에도 CLS, SEP 위치 -100으로 추가
        label_id = [-100] + label_for_tokens + [-100]

        # 패딩
        padding_length = max_length - len(input_id)
        input_id += [tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        label_id += [-100] * padding_length

        # 자르기
        input_ids.append(input_id[:max_length])
        attention_masks.append(attention_mask[:max_length])
        label_ids.append(label_id[:max_length])

    encodings = {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_masks),
        "labels": torch.tensor(label_ids)
    }

    return encodings


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
