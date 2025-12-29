import os
import numpy as np
import torch
from transformers import (AutoModelForTokenClassification, TrainingArguments,
                          DataCollatorForTokenClassification, Trainer)
import evaluate
from utils import set_seed, load_label_list, compute_class_weights, save_json
from data_loader import build_datasets, tokenize_and_align

MODEL_NAME = "skt/kobert-base-v1"
DATA_DIR = "./KORBERT_NER_TYPE/data"
SAVE_DIR = "./KORBERT_NER_TYPE/results"

TRAIN_PATH = f"{DATA_DIR}/train.tsv"
EVAL_PATH  = f"{DATA_DIR}/eval_test.tsv"
TEST_PATH  = f"{DATA_DIR}/test.tsv"

def main():
    set_seed(42)
    labels, label2id, id2label = load_label_list()

    # 데이터 빌드 & 토크나이즈/정렬
    ds = build_datasets(TRAIN_PATH, EVAL_PATH, TEST_PATH)
    tokenized, tokenizer = tokenize_and_align(ds, MODEL_NAME, label2id, max_len=128)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(labels), id2label=id2label, label2id=label2id
    )

    # 클래스 가중치 (O 편향 완화)
    weights = compute_class_weights(tokenized["train"], label2id)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    metric = evaluate.load("seqeval")

    def compute_metrics(p):
        preds, labels_np = p
        preds = np.argmax(preds, axis=2)
        true_preds, true_labels = [], []
        for pred, lab in zip(preds, labels_np):
            cur_p, cur_l = [], []
            for p_i, l_i in zip(pred, lab):
                if l_i != -100:
                    cur_p.append(id2label[p_i]); cur_l.append(id2label[l_i])
            true_preds.append(cur_p); true_labels.append(cur_l)
        res = metric.compute(predictions=true_preds, references=true_labels)
        return {
            "precision": res["overall_precision"],
            "recall": res["overall_recall"],
            "f1": res["overall_f1"],
            "accuracy": res["overall_accuracy"]
        }

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            # 토크나이저에서 제거했지만 방어적으로 한 번 더 제거
            inputs.pop("token_type_ids", None)
            outputs = model(**inputs)
            logits = outputs.logits
            # 매 스텝마다 디바이스 동기화 + -100 무시
            w = weights.to(logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=w, ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    args = TrainingArguments(
        output_dir=SAVE_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{SAVE_DIR}/logs",
        logging_steps=50,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        learning_rate=5e-5,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        report_to="none"
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        data_collator=data_collator,
        tokenizer=tokenizer,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        compute_metrics=compute_metrics
    )

    trainer.train()
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_json(trainer.evaluate(), f"{SAVE_DIR}/eval_metrics.json")
    trainer.save_model(f"{SAVE_DIR}/best_model")
    tokenizer.save_pretrained(f"{SAVE_DIR}/best_model")

    save_json(trainer.evaluate(eval_dataset=tokenized["test"]), f"{SAVE_DIR}/test_metrics.json")

if __name__ == "__main__":
    main()
