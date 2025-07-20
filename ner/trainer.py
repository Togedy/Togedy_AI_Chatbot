import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
from utils import compute_metrics, compute_class_weights

# Load labels
label_file = "./data/label.txt"
with open(label_file, "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f.readlines()]
label_to_id = {label: i for i, label in enumerate(labels)}
id_to_label = {i: label for label, i in label_to_id.items()}
num_labels = len(labels)

# Load tokenizer & model
model_name = "skt/kobert-base-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)

# Load dataset
data_files = {
    "train": "./data/train.tsv",
    "test": "./data/test.tsv"
}
raw_datasets = load_dataset("csv", data_files=data_files, delimiter="\t", quoting=3, column_names=["tokens", "labels"])

# Remove None data
raw_datasets = raw_datasets.filter(lambda example: example["tokens"] is not None and example["labels"] is not None)

# Preprocessing
label_all_tokens = True

def tokenize_and_align_labels(example):
    if example["tokens"] is None or example["labels"] is None:
        return {}

    tokens = example["tokens"].split()
    labels_ = example["labels"].split()

    tokenized_inputs = tokenizer(
        tokens,
        truncation=True,
        padding="max_length",
        max_length=128,
        is_split_into_words=True
    )

    word_ids = tokenized_inputs.word_ids()
    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:
        if word_idx is None or word_idx >= len(labels_):
            label_ids.append(-100)
        elif word_idx != previous_word_idx:
            label_ids.append(label_to_id.get(labels_[word_idx], 0))
        else:
            label_ids.append(label_to_id.get(labels_[word_idx], 0) if label_all_tokens else -100)
        previous_word_idx = word_idx

    tokenized_inputs["labels"] = label_ids
    return tokenized_inputs

# Tokenize dataset
tokenized_datasets = raw_datasets.map(
    tokenize_and_align_labels,
    remove_columns=["tokens", "labels"]
)

# Compute class weights
train_labels = [label for row in raw_datasets["train"] for label in row["labels"].split()]
class_weights = compute_class_weights(labels, {"train": train_labels})
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

# Wrap model with weighted loss and no token_type_ids
class WeightedTokenClassificationModel(torch.nn.Module):
    def __init__(self, base_model, class_weights):
        super().__init__()
        self.base_model = base_model
        self.loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Explicitly remove token_type_ids for KoBERT
        if "token_type_ids" in kwargs:
            kwargs.pop("token_type_ids")

        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        logits = outputs.logits

        if labels is not None:
            loss = self.loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}

# Wrap model
wrapped_model = WeightedTokenClassificationModel(model, class_weights_tensor)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",  # transformers>=4.46부터는 eval_strategy 사용 가능
    save_strategy="epoch",
    logging_dir="./logs",
    logging_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=5e-5,
    weight_decay=0.01,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    remove_unused_columns=False
)

# Trainer
trainer = Trainer(
    model=wrapped_model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train
trainer.train()
