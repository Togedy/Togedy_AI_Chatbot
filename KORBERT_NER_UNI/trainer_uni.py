# trainer_uni.py
import os
import torch
from collections import Counter
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    Trainer, TrainingArguments, DataCollatorForTokenClassification
)
from datasets import DatasetDict
from sklearn.metrics import classification_report
from data_loader import load_dataset
from utils import compute_class_weights, get_label_list, set_seed

# ---- 경로 자동화 ----
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
SAVE_DIR = os.path.join(ROOT, "results")

LABEL_PATH = os.path.join(DATA_DIR, "label.txt")          # [B-UNI, I-UNI, O]
TRAIN_PATH = os.path.join(DATA_DIR, "train.tsv")
TEST_PATH  = os.path.join(DATA_DIR, "eval_test.tsv")
MODEL_CHECKPOINT = "skt/kobert-base-v1"
SAVE_PATH  = SAVE_DIR

def main():
    os.makedirs(SAVE_PATH, exist_ok=True)
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # 1) 라벨
    label_list, label_to_id, id_to_label = get_label_list(LABEL_PATH)
    num_labels = len(label_list)
    print(f"Labels: {label_list}")

    # 2) 토크나이저/모델 (KoBERT: use_fast=False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, use_fast=False)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOINT, num_labels=num_labels
    ).to(device)

    # 3) 데이터 로드
    print("Loading dataset...")
    raw_datasets: DatasetDict = load_dataset(TRAIN_PATH, TEST_PATH, valid_ratio=0.1)
    print({k: len(v) for k, v in raw_datasets.items()})  # train/validation/test

    # 4) 토큰 정렬 (WordPiece / KoBERT 비-패스트 대응)
    def tokenize_and_align_labels(example):
        tokens = example["tokens"]
        labels_ = example["labels"]

        enc = tokenizer(
            tokens,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=128
        )

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        wp_tokens = tokenizer.convert_ids_to_tokens(input_ids)

        label_ids = []
        word_idx = -1
        for tok in wp_tokens:
            if tok in ["[CLS]", "[SEP]", "[PAD]"]:
                label_ids.append(-100)
                continue
            if not tok.startswith("##"):          # 새 단어 시작
                word_idx += 1
            if 0 <= word_idx < len(labels_):
                lab = labels_[word_idx]
                # 서브워드에는 I-로 확장
                if tok.startswith("##") and lab.startswith("B-"):
                    lab = "I-" + lab[2:]
                label_ids.append(label_to_id.get(lab, label_to_id["O"]))
            else:
                label_ids.append(-100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_ids
        }

    tokenized = raw_datasets.map(
        tokenize_and_align_labels,
        remove_columns=["tokens", "labels"]
    )

    # 5) 라벨 분포 로그 (토크나이즈 후)
    lab_cnt = Counter()
    for ex in tokenized["train"]:
        for y in ex["labels"]:
            if y != -100:
                lab_cnt[y] += 1
    _id2lab = {i: l for i, l in enumerate(label_list)}
    print("[Tokenized label stats]:", { _id2lab[i]: c for i, c in lab_cnt.items() })

    data_collator = DataCollatorForTokenClassification(tokenizer)

    # 6) 클래스 가중치 (O 편향 완화)
    class_weights = compute_class_weights(label_list, tokenized["train"]).to(device)

    # 7) 메트릭 (리포트 콘솔 출력)
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        y_true, y_pred = [], []
        for p, l in zip(preds, labels):
            for pi, li in zip(p, l):
                if li != -100:
                    y_true.append(_id2lab[int(li)])
                    y_pred.append(_id2lab[int(pi)])
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, labels=label_list, zero_division=0))
        return {}

    # 8) CustomTrainer: token_type_ids 제거 (KoBERT 안전)
    from torch.nn import CrossEntropyLoss
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            inputs.pop("token_type_ids", None)  # 안전 제거
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = CrossEntropyLoss(weight=class_weights, ignore_index=-100)
            loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    # 9) 학습 설정
    args = TrainingArguments(
        output_dir=SAVE_PATH,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{SAVE_PATH}/logs",
        logging_steps=50,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[]
    )

    trainer = CustomTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # 10) 체크포인트 저장
    ckpt_dir = os.path.join(SAVE_PATH, "uni_best")
    trainer.save_model(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    print("✅ Saved checkpoint to:", ckpt_dir)

if __name__ == "__main__":
    main()
