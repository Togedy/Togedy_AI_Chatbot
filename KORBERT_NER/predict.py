import torch
from transformers import BertTokenizer, AutoModelForTokenClassification
from utils import get_label_list

# 라벨 파일 경로
label_path = "./KORBERT_NER/data/label.txt"
label_list = get_label_list(label_path)
id2label = {i: label for i, label in enumerate(label_list)}

# 저장된 모델 및 vocab 경로
model_path = "./KORBERT_NER/model"
vocab_path = "./KORBERT_NER/tokenizer/vocab.txt"

# ✅ 토크나이저를 직접 vocab으로부터 로드
tokenizer = BertTokenizer(vocab_file=vocab_path)

# 모델 불러오기
model = AutoModelForTokenClassification.from_pretrained(model_path, trust_remote_code=True)
model.eval()

# 예측 함수
def predict(sentence):
    words = sentence.split()
    inputs = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True, padding=True)

    # token_type_ids 제거 (에러 방지용)
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=2)
    labels = [id2label[pred.item()] for pred in predictions[0]]
    return list(zip(words, labels))

# 테스트 실행
if __name__ == "__main__":
    test_sent = "건국대학교 수시 모집 알려줘"
    print(predict(test_sent))
