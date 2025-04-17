from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# 모델과 토크나이저 로드
model_name = "skt/kobert-base-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=2)

# 테스트 문장
sentence = "안녕하세요 홍길동입니다"
tokens = tokenizer.tokenize(sentence)
inputs = tokenizer(sentence, return_tensors="pt")

# Kobert에는 token_type_ids가 필요 없는 경우가 많음
inputs.pop("token_type_ids", None)

# 추론
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)

# 출력
print("입력 토큰:", tokens)
print("예측 결과:", predictions[0].tolist())
