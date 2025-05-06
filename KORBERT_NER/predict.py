import torch
from transformers import BertTokenizer, AutoModelForTokenClassification
from utils import get_label_list
from seqeval.metrics import classification_report
import sys

# 설정
label_path = "./KORBERT_NER/data/label.txt"
label_list = get_label_list(label_path)
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in enumerate(label_list)}

model_path = "./KORBERT_NER/model"
vocab_path = "./KORBERT_NER/tokenizer/vocab.txt"
test_path = "./KORBERT_NER/data/test.tsv"

tokenizer = BertTokenizer(vocab_file=vocab_path)
model = AutoModelForTokenClassification.from_pretrained(model_path, trust_remote_code=True)
model.eval()

def predict(sentence):
    words = sentence.split()
    inputs = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True, padding=True)
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2)
    labels = [id2label[pred.item()] for pred in predictions[0]]
    return list(zip(words, labels))

def evaluate_from_test_file(test_path):
    sentences = []
    true_labels = []
    with open(test_path, encoding="utf-8") as f:
        words, labels = [], []
        for line in f:
            line = line.strip()
            if not line:
                if words:
                    sentences.append(words)
                    true_labels.append(labels)
                    words, labels = [], []
                continue
            splits = line.split('\t')
            if len(splits) != 2:
                continue
            word, label = splits
            words.append(word)
            labels.append(label)
        if words:
            sentences.append(words)
            true_labels.append(labels)

    pred_labels = []
    for words in sentences:
        inputs = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True, padding=True)
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]
        with torch.no_grad():
            outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)[0]
        preds = [id2label[pred.item()] for pred in predictions[:len(words)]]
        pred_labels.append(preds)

    print("=== 평가 결과 ===")
    print(classification_report(true_labels, pred_labels, digits=4))

if __name__ == "__main__":
    mode = "2"
    if mode == "1":
        sentence = "건국대학교 KU논술 전형은 언제야?"
        print(predict(sentence))
    elif mode == "2":
        evaluate_from_test_file(test_path)
    else:
        print("잘못된 모드입니다.")
