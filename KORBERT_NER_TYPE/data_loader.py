from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from typing import Dict

def read_tsv(path: str):
    sents_tokens, sents_labels = [], []
    cur_toks, cur_labs = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if cur_toks:
                    sents_tokens.append(cur_toks); sents_labels.append(cur_labs)
                    cur_toks, cur_labs = [], []
                continue
            tok, lab = line.split("\t")
            cur_toks.append(tok); cur_labs.append(lab)
    if cur_toks:
        sents_tokens.append(cur_toks); sents_labels.append(cur_labs)
    return {"tokens": sents_tokens, "ner_tags": sents_labels}

def build_datasets(train_path, eval_path, test_path):
    train_raw = read_tsv(train_path)
    eval_raw  = read_tsv(eval_path)
    test_raw  = read_tsv(test_path)
    return DatasetDict({
        "train": Dataset.from_dict(train_raw),
        "validation": Dataset.from_dict(eval_raw),
        "test": Dataset.from_dict(test_raw)
    })

def tokenize_and_align(ds: DatasetDict, model_name: str, label2id: Dict[str, int], max_len: int = 128):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def align(batch):
        enc = tokenizer(batch["tokens"], is_split_into_words=True, truncation=True, max_length=max_len)
        aligned_labels = []
        for i in range(len(batch["tokens"])):
            word_ids = enc.word_ids(batch_index=i)
            labs = batch["ner_tags"][i]
            lab_ids, prev_wid = [], None
            for wid in word_ids:
                if wid is None:
                    lab_ids.append(-100)
                else:
                    lab = labs[wid]  # "B-TYPE"/"I-TYPE"/"O"
                    if wid != prev_wid:
                        lab_ids.append(label2id.get(lab, label2id["O"]))
                    else:
                        lab_ids.append(label2id.get(lab.replace("B-", "I-"), label2id["O"]))
                prev_wid = wid
            aligned_labels.append(lab_ids)
        enc["labels"] = aligned_labels
        # KoBERT 안전: token_type_ids 제거
        if "token_type_ids" in enc:
            del enc["token_type_ids"]
        return enc

    tokenized = ds.map(align, batched=True, remove_columns=["tokens", "ner_tags"])
    return tokenized, tokenizer
