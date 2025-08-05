import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import Dataset
from seqeval.metrics import classification_report
from data_loader import read_tsv_file
from utils import get_label_list
from postprocess import postprocess_ner_output

LABEL_PATH = "./data/label.txt"
MODEL_PATH = "./results/checkpoint-8439"
MODEL_NAME = "skt/kobert-base-v1"
TEST_PATH = "./data/test.tsv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# 1. Load label info
label_list, label_to_id, id_to_label = get_label_list(LABEL_PATH)

# 2. Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH, num_labels=len(label_list)).to(device)
model.eval()
print("✅ Model and tokenizer loaded")

# 3. Load test data
test_data = read_tsv_file(TEST_PATH)
test_dataset = Dataset.from_dict(test_data)
print(f"✅ Loaded test samples: {len(test_dataset)}")

# 4. Tokenization with alignment
def tokenize_and_align_labels(example):
    tokens = example["tokens"]
    labels_ = example["labels"]

    tokenized_inputs = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

    input_ids = tokenized_inputs["input_ids"][0]
    attention_mask = tokenized_inputs["attention_mask"][0]

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
tokenized = test_dataset.map(tokenize_and_align_labels)

# 5. Evaluation loop
all_preds = []
all_labels = []

for item in tokenized:
    input_ids = torch.tensor([item["input_ids"]]).to(device)
    attention_mask = torch.tensor([item["attention_mask"]]).to(device)
    labels = item["labels"]

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=2).squeeze().tolist()

    pred_tags = []
    true_tags = []

    for p, l in zip(preds, labels):
        if l != -100:
            pred_tags.append(id_to_label[p])
            true_tags.append(id_to_label[l])

    all_preds.append(pred_tags)
    all_labels.append(true_tags)

# 6. Print classification report
print("\n📊 Classification Report:")
print(classification_report(all_labels, all_preds))
