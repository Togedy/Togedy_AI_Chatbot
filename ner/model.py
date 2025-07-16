# ner/model.py

from transformers import BertTokenizerFast, BertForTokenClassification
import torch

MODEL_SAVE_PATH = "./ner/model_save"

def load_ner_model():
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_SAVE_PATH)
    model = BertForTokenClassification.from_pretrained(MODEL_SAVE_PATH)
    return tokenizer, model

def predict_ner(model_bundle, sentence: str):
    tokenizer, model = model_bundle
    model.eval()
    tokens = tokenizer.tokenize(sentence)
    inputs = tokenizer(sentence, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()
    label_list = open("./data/label.txt").read().splitlines()
    labels = [label_list[p] for p in predictions[1:len(tokens)+1]]  # remove CLS, SEP
    return tokens, labels
