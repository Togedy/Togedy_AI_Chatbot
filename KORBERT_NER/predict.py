import torch
from transformers import BertTokenizer, BertForTokenClassification
from utils import get_label_list

label_path = "./KORBERT_NER/data/label.txt"
label_list = get_label_list(label_path)
id2label = {i: label for i, label in enumerate(label_list)}

model_name = "skt/kobert-base-v1"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForTokenClassification.from_pretrained(model_name, num_labels=len(label_list))
model.eval()

def predict(sentence):
    tokens = sentence.split()
    inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2)
    labels = [id2label[pred.item()] for pred in predictions[0]]
    return list(zip(tokens, labels))

if __name__ == "__main__":
    sentence = "건국대학교 수시 모집 알려줘"
    print(predict(sentence))
