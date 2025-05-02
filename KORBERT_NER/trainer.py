import torch
from transformers import BertForTokenClassification, Trainer, TrainingArguments, AutoTokenizer
from data_loader import load_and_cache_examples
from utils import compute_metrics, get_label_list

MODEL_NAME = "skt/kobert-base-v1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
label_list = get_label_list("./KORBERT_NER/data/label.txt")
num_labels = len(label_list)

train_dataset = load_and_cache_examples(tokenizer, label_list, mode="train")
eval_dataset = load_and_cache_examples(tokenizer, label_list, mode="test")

model = BertForTokenClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

training_args = TrainingArguments(
    output_dir="./KORBERT_NER/results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_dir="./KORBERT_NER/logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
