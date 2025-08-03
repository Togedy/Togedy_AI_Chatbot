import csv
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.metrics import classification_report
from utils import get_label_list

LABEL_PATH = "./data/label.txt"
MODEL_NAME = "skt/kobert-base-v1"
MODEL_PATH = "./results/checkpoint-1680"
TEST_PATH = "./data/test.tsv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# Load label info
label_list, label_to_id, id_to_label = get_label_list(LABEL_PATH)
num_labels = len(label_list)

# Load model & tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH, num_labels=num_labels).to(device)
print("✅ Model and tokenizer loaded")

# Read test.tsv directly (문장 단위로 분리)
def read_tsv_file(file_path):
    data = []
    with open(file_path, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        tokens, labels = [], []
        for line in reader:
            if not line or line[0].strip() == "":
                if tokens:
                    data.append({"tokens": tokens, "labels": labels})
                    tokens, labels = [], []
            else:
                token, label = line
                tokens.append(token)
                labels.append(label)
        if tokens:
            data.append({"tokens": tokens, "labels": labels})
    return data

print("📂 Loading test dataset...")
test_examples = read_tsv_file(TEST_PATH)
test_dataset = Dataset.from_list(test_examples)

# Tokenization
def tokenize_and_align_labels(example):
    tokens = example["tokens"]
    labels_ = example["labels"]

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
        "labels": label_ids
    }

print("🔄 Tokenizing test data...")
tokenized_test = test_dataset.map(tokenize_and_align_labels)

# Evaluation
print("🚀 Running evaluation...")
model.eval()
true_labels = []
true_predictions = []

for batch in tokenized_test:
    input_ids = torch.tensor([batch["input_ids"]]).to(device)
    attention_mask = torch.tensor([batch["attention_mask"]]).to(device)
    labels = batch["labels"]

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2)[0].tolist()

    for p, l in zip(predictions, labels):
        if l != -100:
            true_labels.append(id_to_label[l])
            true_predictions.append(id_to_label[p])

print("\n📊 Classification Report:")
print(classification_report(true_labels, true_predictions, digits=4))
