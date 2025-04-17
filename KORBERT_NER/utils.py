import os
import logging
import torch
from transformers import (
    BertConfig,
    BertForTokenClassification,
    AutoTokenizer,
    ElectraConfig,
    ElectraForTokenClassification,
    DistilBertConfig,
    DistilBertForTokenClassification,
    BertTokenizer
)
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'kobert': (BertConfig, BertForTokenClassification, AutoTokenizer),
    'distilkobert': (DistilBertConfig, DistilBertForTokenClassification, AutoTokenizer),
    'bert': (BertConfig, BertForTokenClassification, BertTokenizer),
    'kobert-lm': (BertConfig, BertForTokenClassification, AutoTokenizer),
    'koelectra-base': (ElectraConfig, ElectraForTokenClassification, AutoTokenizer),
    'koelectra-small': (ElectraConfig, ElectraForTokenClassification, AutoTokenizer),
}

def get_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    labels = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    if "O" not in labels:
        labels.append("O")
    return labels

def load_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    return tokenizer

def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return {
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds)
    }

def show_report(preds, labels):
    assert len(preds) == len(labels)
    return classification_report(labels, preds)

def get_intent_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    labels = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    return labels

def get_logger(name, log_path):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s -   %(message)s')

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger
