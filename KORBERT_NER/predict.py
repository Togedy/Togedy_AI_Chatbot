from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from utils import get_label_list

model_name = "skt/kobert-base-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForTokenClassification.from_pretrained("./results/checkpoint-XXX")
model.eval()

label_list = get_label_list('./KORBERT_NER/data/label.txt')
id2label = {i: label for i, label in enumerate(label_list)}

def predict_ner(sentence):
    inputs = tokenizer([list(sentence)], is_split_into_words=True, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs).logits
    predictions = torch.argmax(outputs, dim=2)
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    results = []
    for token, pred in zip(tokens, predictions[0]):
        label = id2label[pred.item()]
        results.append((token, label))
    return results
