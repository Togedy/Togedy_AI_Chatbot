import torch
from transformers import BertTokenizer, BertForTokenClassification, TrainingArguments, Trainer
from data_loader import load_and_cache_examples
from utils import compute_class_weights, compute_metrics, get_label_list
import os

model_name = "skt/kobert-base-v1"
label_list = get_label_list("./KORBERT_NER/data/label.txt")
num_labels = len(label_list)

tokenizer = BertTokenizer.from_pretrained(model_name)
train_dataset = load_and_cache_examples(tokenizer, label_list, mode="train")
eval_dataset = load_and_cache_examples(tokenizer, label_list, mode="test")

class_weights = compute_class_weights(train_dataset, num_labels).to(torch.float)

model = BertForTokenClassification.from_pretrained(
    model_name,
    num_labels=num_labels
)

# Custom Trainer with weighted loss
from torch.nn import CrossEntropyLoss

def compute_loss(model, inputs, return_outputs=False):
    labels = inputs.pop("labels")
    # Remove token_type_ids
    if "token_type_ids" in inputs:
        inputs.pop("token_type_ids")
    outputs = model(**inputs)
    logits = outputs.logits
    loss_fct = CrossEntropyLoss(weight=class_weights, ignore_index=-100)
    loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
    return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.compute_loss = compute_loss

trainer.train()
