# ner/test.py

from model import load_ner_model, predict_ner
from postprocess import postprocess_ner_output
from seqeval.metrics import classification_report, f1_score, accuracy_score
import os

def test_single_sentence():
    sentence = "서울대 수시 모집인원과 연세대 정시 모집 일정이 궁금합니다."

    model = load_ner_model()
    tokens, tags = predict_ner(model, sentence)

    print("[NER 결과]")
    for token, tag in zip(tokens, tags):
        print(f"{token}\t{tag}")

    structured = postprocess_ner_output(sentence, tags, tokens)
    print("\n[후처리된 NER 구조]")
    print(structured)


def test_from_tsv(filepath="./data/test.tsv", evaluate=True):
    print(f"\n[테스트 파일 처리: {filepath}]")

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    model = load_ner_model()

    sentences = []
    current_tokens = []
    current_labels = []

    tokens_list = []  # 전체 토큰 시퀀스
    true_tags_list = []  # 정답
    pred_tags_list = []  # 예측

    for line in lines:
        if line.strip() == "":
            if current_tokens:
                sentence = "".join(current_tokens)
                sentences.append((sentence, current_tokens.copy(), current_labels.copy()))
                current_tokens.clear()
                current_labels.clear()
        else:
            token, label = line.strip().split()
            current_tokens.append(token)
            current_labels.append(label)

    if current_tokens:
        sentence = "".join(current_tokens)
        sentences.append((sentence, current_tokens, current_labels))

    for i, (sentence, tokens_true, tags_true) in enumerate(sentences):
        print(f"\n[{i+1}] 문장: {sentence}")
        tokens_pred, tags_pred = predict_ner(model, sentence)

        # 정렬: 실제 토큰 기준으로 매칭
        print("  토큰별 예측:")
        for t, p, l in zip(tokens_pred, tags_pred, tags_true):
            print(f"    {t}\t→ 예측: {p}\t정답: {l}")

        tokens_list.append(tokens_pred)
        true_tags_list.append(tags_true)
        pred_tags_list.append(tags_pred)

        structured = postprocess_ner_output(sentence, tags_pred, tokens_pred)
        print("  → 추출된 키워드:", structured)

    if evaluate:
        print("\n[정량 평가 지표 (전체)]")
        print("Accuracy:", accuracy_score(true_tags_list, pred_tags_list))
        print("F1 Score:", f1_score(true_tags_list, pred_tags_list))
        print("\nDetailed Report:\n")
        print(classification_report(true_tags_list, pred_tags_list))


if __name__ == "__main__":
    print("▶ 예시 문장 테스트")
    test_single_sentence()

    print("\n\n▶ TSV 기반 테스트 및 평가")
    test_from_tsv()
