import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from utils import load_label
from postprocess import postprocess_ner_output

LABEL_PATH = "./data/label.txt"
MODEL_NAME = "skt/kobert-base-v1"
MODEL_PATH = "./results"

def load_ner_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)  # KoBERT는 fast tokenizer 없음
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
    id_to_label = load_label(LABEL_PATH)
    return tokenizer, model, id_to_label

def predict(sentence, tokenizer, model, id_to_label):
    words = sentence.strip().split()  # 사용자 문장 → 단어 리스트

    inputs = tokenizer(words, return_tensors="pt", is_split_into_words=True, truncation=True)

    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    with torch.no_grad():
        outputs = model(**inputs).logits

    predictions = torch.argmax(outputs, dim=2)[0].tolist()
    input_ids = inputs["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # 수동으로 word-level align
    aligned_tokens = []
    aligned_tags = []

    token_idx = 0
    word_idx = 0
    while token_idx < len(tokens) and word_idx < len(words):
        token = tokens[token_idx]
        label = id_to_label[predictions[token_idx]]

        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            token_idx += 1
            continue

        # SentencePiece tokenizer의 특성: ▁는 단어 경계 표시
        token_clean = token.replace("▁", "")
        if token_clean == "":
            token_idx += 1
            continue

        aligned_tokens.append(words[word_idx])
        aligned_tags.append(label)
        word_idx += 1
        token_idx += 1

    return aligned_tokens, aligned_tags

def test_single_sentence():
    print("▶ 예시 문장 테스트")
    sentence = "한양대 수시 전형 일정 알려줘"
    tokenizer, model, id_to_label = load_ner_model()
    tokens, tags = predict(sentence, tokenizer, model, id_to_label)
    print("Tokens:", tokens)
    print("Tags:", tags)
    result = postprocess_ner_output(sentence, tags, tokens)
    print("후처리 결과:", result)

if __name__ == "__main__":
    test_single_sentence()
