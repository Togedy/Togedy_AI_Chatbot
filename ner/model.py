# ner/model.py

from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

MODEL_SAVE_PATH = "./results"

def load_ner_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_PATH)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_SAVE_PATH)
    return tokenizer, model

def predict_ner(model_bundle, sentence: str):
    tokenizer, model = model_bundle
    model.eval()

    # GPU 사용 가능 시 모델을 GPU에 올림
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tokens = tokenizer.tokenize(sentence)
    inputs = tokenizer(sentence, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()

    # 예측 결과를 라벨로 변환
    label_list = open("./data/label.txt", encoding="utf-8").read().splitlines()
    labels = [label_list[p] for p in predictions[1:len(tokens)+1]]  # [CLS], [SEP] 제거

    return tokens, labels
