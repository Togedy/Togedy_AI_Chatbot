import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification
)
from datasets import DatasetDict
from sklearn.metrics import classification_report
from data_loader import load_dataset
from utils import compute_class_weights, get_label_list

LABEL_PATH = "./data/label.txt"
MODEL_CHECKPOINT = "skt/kobert-base-v1"
TRAIN_PATH = "./data/train.tsv"
TEST_PATH = "./data/test.tsv"
SAVE_PATH = "./results"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n✅ Using device: {device}")

label_list, label_to_id, id_to_label = get_label_list(LABEL_PATH)
print(f"✅ Loaded {len(label_list)} labels")

tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, use_fast=False)
print(f"✅ Tokenizer loaded (Fast: {tokenizer.is_fast})")

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=len(label_list),
)
model.to(device)
print("✅ Model loaded")

print("📂 Loading dataset...")
raw_datasets = load_dataset(TRAIN_PATH, TEST_PATH, valid_ratio=0.1)
print(f"✅ Dataset sizes: Train={len(raw_datasets['train'])}, Val={len(raw_datasets['validation'])}, Test={len(raw_datasets['test'])}")

def tokenize_and_align_labels(example):
    tokens = example["tokens"]
    labels_ = example["labels"]

    if len(tokens) != len(labels_):
        print(f"⚠️ Token-label length mismatch: tokens={len(tokens)} labels={len(labels_)}")
        return {"input_ids": [], "attention_mask": [], "labels": [], "ignore": True}

    tokenized_inputs = tokenizer(
        tokens,
        is_split_into_words=True,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    input_ids = tokenized_inputs["input_ids"][0].tolist()
    attention_mask = tokenized_inputs["attention_mask"][0].tolist()

    label_ids = []
    token_index = 0
    for token in tokenizer.convert_ids_to_tokens(input_ids):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            label_ids.append(-100)
        elif token.startswith("##"):
            label_ids.append(label_to_id.get(labels_[token_index - 1], label_to_id["O"]))
        else:
            if token_index < len(labels_):
                label_ids.append(label_to_id.get(labels_[token_index], label_to_id["O"]))
                token_index += 1
            else:
                label_ids.append(-100)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": label_ids,
        "ignore": False
    }

print("🔄 Tokenizing and aligning labels...")
tokenized_dict = raw_datasets.map(tokenize_and_align_labels, remove_columns=["tokens", "labels"])
filtered_dict = {k: v.filter(lambda x: not x["ignore"]) for k, v in tokenized_dict.items()}
tokenized_datasets = DatasetDict(filtered_dict)

data_collator = DataCollatorForTokenClassification(tokenizer)

# ✅ 수정된 부분
class_weights = compute_class_weights(label_list, tokenized_datasets["train"]).to(device)

def compute_metrics(pred):
    predictions, labels = pred
    predictions = predictions.argmax(axis=-1)
    true_labels, true_predictions = [], []
    for prediction, label in zip(predictions, labels):
        for p, l in zip(prediction, label):
            if l != -100:
                true_labels.append(id_to_label[l])
                true_predictions.append(id_to_label[p])
    print("\n📊 Classification Report:")
    print(classification_report(true_labels, true_predictions))
    return {}

from torch.nn import CrossEntropyLoss
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = CrossEntropyLoss(weight=class_weights, ignore_index=-100)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir=SAVE_PATH,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir=f"{SAVE_PATH}/logs",
    logging_steps=50,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
