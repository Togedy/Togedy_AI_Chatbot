import torch
from transformers import BertForTokenClassification, BertTokenizer, Trainer, TrainingArguments
from utils import get_label_list, compute_metrics
from data_loader import load_and_cache_examples

# 모델명과 라벨 경로
model_name = "skt/kobert-base-v1"
label_path = "./KORBERT_NER/data/label.txt"

# 라벨 리스트 및 tokenizer
label_list = get_label_list(label_path)
tokenizer = BertTokenizer.from_pretrained(model_name)

# 데이터셋 로드
train_dataset = load_and_cache_examples(tokenizer, label_list, mode="train")

# 모델 준비
model = BertForTokenClassification.from_pretrained(model_name, num_labels=len(label_list))

# 학습 인자
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    evaluation_strategy="no"
)

# Trainer 구성 (eval_dataset 없이)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 학습 수행
trainer.train()
