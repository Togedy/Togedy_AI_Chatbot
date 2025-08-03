import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from utils import load_label
from postprocess import postprocess_ner_output

LABEL_PATH = "./data/label.txt"
MODEL_NAME = "skt/kobert-base-v1"
MODEL_PATH = "./results/checkpoint-1680"

def load_ner_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)  # KoBERT는 fast tokenizer 없음
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
    id_to_label = load_label(LABEL_PATH)
    return tokenizer, model, id_to_label

def predict(sentence, tokenizer, model, id_to_label):
    words = sentence.strip().split()

    inputs = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )

    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs).logits

    predictions = torch.argmax(outputs, dim=2)[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    aligned_tokens = []
    aligned_tags = []

    word_idx = 0
    for idx, token in enumerate(tokens):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        label = id_to_label[predictions[idx]]

        # sentencepiece에서 ▁는 공백(단어 시작) → 원 단어 단위 정렬 시 사용
        if word_idx < len(words):
            aligned_tokens.append(words[word_idx])
            aligned_tags.append(label)
            word_idx += 1

    return aligned_tokens, aligned_tags

def test_single_sentence():
    print("▶ 예시 문장 테스트")
    sentence = "한양대 건국대, 수시 전형 일정 알려줘"
    tokenizer, model, id_to_label = load_ner_model()
    tokens, tags = predict(sentence, tokenizer, model, id_to_label)
    print("Tokens:", tokens)
    print("Tags:", tags)
    result = postprocess_ner_output(sentence, tags, tokens)  # 인자 순서: 문장, 태그, 토큰
    print("후처리 결과:", result)

if __name__ == "__main__":
    test_single_sentence()
