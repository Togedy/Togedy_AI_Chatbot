# ner/trainer.py

from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset, ClassLabel
import os
import numpy as np
from sklearn.metrics import classification_report
import torch

LABEL_PATH = "./data/label.txt"
MODEL_SAVE_PATH = "./ner/model_save"
PRETRAINED_MODEL = "skt/kobert-base-v1"

def load_labels():
    with open(LABEL_PATH, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=2)
    label_list = load_labels()

    true_labels = [[label_list[l] for l in sent] for sent in labels]
    pred_labels = [[label_list[p] for p in sent] for sent in preds]

    report = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)
    return {
        "precision": report["micro avg"]["precision"],
        "recall": report["micro avg"]["recall"],
        "f1": report["micro avg"]["f1-score"],
    }

def tokenize_and_align_labels(example, tokenizer, label2id):
    tokens = []
    label_ids = []

    for word, label in zip(example["tokens"], example["ner_tags"]):
        word_tokens = tokenizer.tokenize(word)
        tokens.extend(word_tokens)
        label_ids.extend([label2id[label]] + [-100] * (len(word_tokens) - 1))

    return tokenizer(" ".join(example["tokens"]), truncation=True, padding="max_length", max_length=128,
                     is_split_into_words=True, return_tensors="pt"), label_ids

def train():
    # 데이터셋 준비
    label_list = load_labels()
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}

    dataset = load_dataset("csv", data_files={"train": "./data/train.tsv", "test": "./data/test.tsv"}, delimiter="\t")
    dataset = dataset.map(lambda example: {"tokens": example["token"].split(), "ner_tags": example["label"].split()})

    tokenizer = BertTokenizerFast.from_pretrained(PRETRAINED_MODEL)
    model = BertForTokenClassification.from_pretrained(PRETRAINED_MODEL, num_labels=len(label_list),
                                                       id2label=id2label, label2id=label2id)

    def preprocess(example):
        encoding = tokenizer(example["tokens"], is_split_into_words=True, padding="max_length",
                             truncation=True, max_length=128)
        labels = []
        word_ids = encoding.word_ids()
        for word_idx in word_ids:
            if word_idx is None:
                labels.append(-100)
            else:
                labels.append(label2id[example["ner_tags"][word_idx]])
        encoding["labels"] = labels
        return encoding

    tokenized_dataset = dataset.map(preprocess, batched=False)

    args = TrainingArguments(
        output_dir="./ner/results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./ner/logs",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=1,
        load_best_model_at_end=True,
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # 저장
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    print(f"모델 저장 완료: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
