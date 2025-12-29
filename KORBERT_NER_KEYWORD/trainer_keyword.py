# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)
from sklearn.metrics import classification_report, precision_recall_fscore_support
from utils import load_label_list, compute_class_weights
from data_loader import KeywordNERDataset, MODEL_CKPT

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DATA_DIR = "./KORBERT_NER_KEYWORD/data"
TRAIN_PATH = os.path.join(DATA_DIR, "train.tsv")
EVAL_PATH  = os.path.join(DATA_DIR, "eval.tsv")
LABEL_PATH = os.path.join(DATA_DIR, "label.txt")
OUTPUT_DIR = "./KORBERT_NER_KEYWORD/results_keyword"

def compute_metrics(p, id2label):
    preds = np.argmax(p.predictions, axis=-1)
    labels = p.label_ids
    # ignore_index=-100 제외하고 계산
    true_labels, true_preds = [], []
    for pred_row, lab_row in zip(preds, labels):
        for p_i, l_i in zip(pred_row, lab_row):
            if l_i == -100:
                continue
            true_labels.append(id2label[l_i])
            true_preds.append(id2label[p_i])
    pr, rc, f1, _ = precision_recall_fscore_support(true_labels, true_preds, average="micro", zero_division=0)
    return {"precision": pr, "recall": rc, "f1": f1}

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    labels, label2id, id2label = load_label_list(LABEL_PATH)
    num_labels = len(labels)

    train_ds = KeywordNERDataset(TRAIN_PATH, label2id)
    eval_ds  = KeywordNERDataset(EVAL_PATH,  label2id)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_CKPT,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    # 클래스 가중치 계산(O 과다 문제 완화)
    class_weights = compute_class_weights(
        all_label_ids=[ex["labels"].tolist() for ex in train_ds],
        num_labels=num_labels,
        ignore_index=-100
    ).to(device)

    # Trainer에 커스텀 loss 적용
    def custom_loss(model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
        loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        if return_outputs:
            return loss, outputs
        return loss

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        evaluation_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        metric_for_best_model="f1",
        load_best_model_at_end=True,
        report_to="none",
        label_smoothing_factor=0.0,
        logging_steps=50,
        seed=42
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=train_ds.tok)

    # Trainer 재정의: compute_loss 오버라이드
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            return custom_loss(model, inputs, return_outputs)

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=train_ds.tok,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, id2label),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    trainer.train()

    # 평가 리포트 출력
    preds = trainer.predict(eval_ds)
    y_pred = np.argmax(preds.predictions, axis=-1)
    y_true = preds.label_ids

    true_labels, true_preds = [], []
    for pred_row, lab_row in zip(y_pred, y_true):
        for p_i, l_i in zip(pred_row, lab_row):
            if l_i == -100:
                continue
            true_labels.append(id2label[l_i])
            true_preds.append(id2label[p_i])

    print("\nClassification Report (KEYWORD NER):")
    print(classification_report(true_labels, true_preds, labels=labels, digits=4, zero_division=0))

    trainer.save_model(OUTPUT_DIR)
    train_ds.tok.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
