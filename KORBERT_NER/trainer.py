from transformers import BertForTokenClassification, Trainer, TrainingArguments, AutoTokenizer
from data_loader import load_and_cache_examples
from utils import get_label_list, compute_metrics

MODEL_NAME = "skt/kobert-base-v1"
LABEL_PATH = "./KORBERT_NER/data/label.txt"
OUTPUT_DIR = "./kobert-ner-model"

label_list = get_label_list(LABEL_PATH)
num_labels = len(label_list)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = BertForTokenClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

train_dataset = load_and_cache_examples(tokenizer, label_list, mode="train")
eval_dataset = load_and_cache_examples(tokenizer, label_list, mode="test")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    learning_rate=5e-5,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=lambda p: compute_metrics(p, label_list)
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
