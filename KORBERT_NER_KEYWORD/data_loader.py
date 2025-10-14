# -*- coding: utf-8 -*-
from typing import List, Tuple, Dict
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# KoBERT fast tokenizer 사용 필수(워드-서브워드 정렬에 필요)
MODEL_CKPT = "skt/kobert-base-v1"

def read_tsv_sequence(path: str) -> List[Tuple[List[str], List[str]]]:
    """
    문장 간 빈 줄 구분. 각 줄: token \t label
    반환: [(tokens, labels), ...]
    """
    samples, tokens, labels = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if tokens:
                    samples.append((tokens, labels))
                    tokens, labels = [], []
                continue
            tok, lab = line.split("\t")
            tokens.append(tok)
            labels.append(lab)
    if tokens:
        samples.append((tokens, labels))
    return samples

@dataclass
class TokenizedBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor

class KeywordNERDataset(Dataset):
    def __init__(self, path: str, label2id: Dict[str, int], max_length: int = 256):
        self.samples = read_tsv_sequence(path)
        self.label2id = label2id
        self.max_length = max_length
        # fast tokenizer 필요
        self.tok = AutoTokenizer.from_pretrained(MODEL_CKPT, use_fast=True)

        self.encodings = []
        self.label_ids = []

        for tokens, tags in self.samples:
            enc = self.tok(
                tokens,
                is_split_into_words=True,
                truncation=True,
                max_length=self.max_length,
                return_offsets_mapping=False
            )
            # 워드-서브워드 정렬
            word_ids = enc.word_ids()
            aligned = []
            prev_word_id = None
            for w_id in word_ids:
                if w_id is None:
                    aligned.append(-100)  # special tokens
                else:
                    lab = tags[w_id]
                    if w_id != prev_word_id:
                        aligned.append(self.label2id.get(lab, self.label2id["O"]))
                    else:
                        # 동일 단어의 서브토큰: I-KEYWORD 규칙 적용
                        if lab.startswith("B-KEYWORD"):
                            aligned.append(self.label2id.get("I-KEYWORD", self.label2id["O"]))
                        else:
                            aligned.append(self.label2id.get(lab, self.label2id["O"]))
                prev_word_id = w_id

            self.encodings.append(enc)
            self.label_ids.append(aligned)

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v) for k, v in self.encodings[idx].items()}
        labels = torch.tensor(self.label_ids[idx])
        # 안전: token_type_ids 제거(모델이 기대하지 않아도 들어갈 수 있음)
        item.pop("token_type_ids", None)
        item["labels"] = labels
        return item
