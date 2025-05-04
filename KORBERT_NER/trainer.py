import os
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from utils import get_label_list, compute_metrics
from data_loader import load_and_cache_examples

# 모델명과 라벨 경로
model_name = "monologg/kobert"
label_path = "./KORBERT_NER/data/label.txt"

# 라벨 리스트 및 tokenizer
label_list = get_label_list(label_path)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)

# 데이터셋 로드
train_dataset = load_and_cache_examples(tokenizer, label_list, mode="train")

# 모델 준비
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(label_list), trust_remote_code=True)

# 학습 인자
training_args = TrainingArguments(
    output_dir="./KORBERT_NER/model",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=0,
    evaluation_strategy="no"
)

# Trainer 구성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=None,  # tokenizer 저장은 직접 수행하므로 None
    compute_metrics=compute_metrics
)

# 학습 수행
trainer.train()

# 모델 저장
model.save_pretrained("./KORBERT_NER/model")

# ✅ tokenizer 저장: save_vocabulary만 사용
vocab_path = os.path.join("./KORBERT_NER/tokenizer")
os.makedirs(vocab_path, exist_ok=True)
tokenizer.save_vocabulary(vocab_path)
