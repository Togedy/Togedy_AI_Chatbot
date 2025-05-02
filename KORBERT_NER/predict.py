import torch
from transformers import AutoTokenizer, BertForTokenClassification
from utils import get_label_list

MODEL_PATH = "./kobert-ner-model"
LABEL_PATH = "./KORBERT_NER/data/label.txt"

label_list = get_label_list(LABEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
model = BertForTokenClassification.from_pretrained(MODEL_PATH)
model.eval()

def predict_ner(sentence):
    # 문장을 공백 기준으로 split → word 단위 예측
    words = sentence.strip().split()
    encoding = tokenizer(words, is_split_into_words=True, return_tensors="pt", return_offsets_mapping=True, truncation=True, max_length=512)
    word_ids = encoding.word_ids()

    encoding.pop("token_type_ids", None)
    encoding.pop("offset_mapping", None)

    with torch.no_grad():
        outputs = model(**encoding)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)[0].tolist()

    # word_id -> word 단위 재정렬
    results = []
    prev_word_idx = None
    word_result = None

    for idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        if word_idx != prev_word_idx:
            word_result = (words[word_idx], label_list[predictions[idx]])
            results.append(word_result)
            prev_word_idx = word_idx

    return results
