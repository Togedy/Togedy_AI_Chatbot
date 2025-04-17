import os
import torch
import numpy as np
from datasets import load_metric, Dataset
from transformers import (
    BertForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    AutoTokenizer,
    AutoConfig
)
from sklearn.model_selection import train_test_split
from utils import read_data, encode_examples, get_label_list

# 설정
MODEL_NAME = "skt/kobert-base-v1"
LABEL_FILE = "label.txt"
DATA_FILE = "train_data.txt"  # 각 문장이 BIO 형식으로 라벨링된 파일
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 5
OUTPUT_DIR = "./kobert-ner-model"

# 1. 라벨 준비
label_list = get_label_list(LABEL_FILE)
label_to_id = {label: i for i, label in enumerate(label_list)}
id_to_label = {i: label for label, i in label_to_id.items()}

# 2. 데이터 준비
examples = read_data(DATA_FILE)
train_ex, val_ex = train_test_split(examples, test_size=0.1, random_state=42)

# 3. Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 4. Encoding
train_dataset = encode_examples(train_ex, tokenizer, label_to_id, MAX_LEN)
val_dataset = encode_examples(val_ex, tokenizer, label_to_id, MAX_LEN)

# 5. Hugging Face Dataset 변환
train_dataset = Dataset.from_dict(train_dataset)
val_dataset = Dataset.from_dict(val_dataset)

# 6. 모델 구성
config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=len(label_list), id2label=id_to_label, label2id=label_to_id)
model = BertForTokenClassification.from_pretrained(MODEL_NAME, config=config)

# 7. Metric 정의
metric = load_metric("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [id_to_label[p] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    true_labels = [
        [id_to_label[l] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# 8. Trainer 설정
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    logging_dir=f"{OUTPUT_DIR}/logs",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

data_collator = DataCollatorForTokenClassification(tokenizer)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 9. 학습 시작
trainer.train()

# 10. 저장
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
