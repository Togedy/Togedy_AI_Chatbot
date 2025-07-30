# ner/postprocess.py

from typing import List, Dict

def postprocess_ner_output(sentence: str, ner_tags: List[str], tokens: List[str]) -> Dict[str, List[str]]:
    """
    BIO 태그 시퀀스에서 UNI, TYPE, KEYWORD 엔티티를 추출하여 딕셔너리로 반환합니다.
    :param sentence: 원문 질문 (예: "서울대 수시 모집 인원 알려줘")
    :param ner_tags: BIO 형식의 예측 결과 (예: ["B-UNI", "B-TYPE", "B-KEYWORD", ...])
    :param tokens: tokenizer로 분리된 토큰 (예: ["서울대", "수시", "모집", "인원", ...])
    :return: {"UNI": [...], "TYPE": [...], "KEYWORD": [...]}
    """
    entities = {"UNI": [], "TYPE": [], "KEYWORD": []}
    current_entity = None
    current_tokens = []

    for tag, token in zip(ner_tags, tokens):
        if tag == "O":
            if current_entity:
                entities[current_entity].append("".join(current_tokens))
                current_entity, current_tokens = None, []
        elif tag.startswith("B-"):
            if current_entity:
                entities[current_entity].append("".join(current_tokens))
            current_entity = tag[2:]
            current_tokens = [token]
        elif tag.startswith("I-") and current_entity == tag[2:]:
            current_tokens.append(token)
        else:
            if current_entity:
                entities[current_entity].append("".join(current_tokens))
            current_entity, current_tokens = None, []

    # 마지막 엔티티가 남아있는 경우
    if current_entity and current_tokens:
        entities[current_entity].append("".join(current_tokens))

    return entities
