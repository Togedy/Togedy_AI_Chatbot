def build_prompt(query: dict, retrieved_docs: list) -> str:
    """
    GPT 입력용 프롬프트 생성
    """
    lines = ["[입시 정보]"]

    for doc in retrieved_docs:
        univ_kr = uni_name_kr(doc["university"])
        page_info = f"{univ_kr} {doc['source'].replace('_text.txt', '')} p.{doc['page']}"
        lines.append(f"\n출처: {page_info}")
        lines.append(f"내용: {doc['text'].strip()[:1000]}")  # 길이 제한

    # 사용자 질문
    user_question = build_user_question(query)
    lines.append(f"\n질문: {user_question}")
    lines.append("→ 위 내용을 참고하여 정확하고 간결하게 요약해 주세요.")

    return "\n".join(lines)


def build_user_question(query: dict) -> str:
    """
    GPT에게 던질 사용자 질문 생성
    (형식이나 문장을 이 함수에서 제어)

    Args:
        query: {
            "UNI": "yonsei",
            "TYPE": "susi",
            "KEYWORD": "논술"
        }

    Returns:
        str: 생성된 질문 문장
    """
    univ = uni_name_kr(query.get("UNI", ""))
    type_ = query.get("TYPE", "")
    keyword = query.get("KEYWORD", "")

    if univ and keyword and type_:
        return f"{univ}의 {type_} 전형 중 '{keyword}'에 대해 알려줘"
    elif univ and keyword:
        return f"{univ}의 '{keyword}'에 대해 알려줘"
    elif keyword:
        return f"'{keyword}'에 대해 알려줘"
    else:
        return "입시 관련 정보를 알려줘"


def uni_name_kr(eng_name: str) -> str:
    mapping = {
        "yonsei": "연세대학교",
        "korea": "고려대학교",
        "sogang": "서강대학교",
        "skku": "성균관대학교",
        "hanyang": "한양대학교",
        "seoul": "서울대학교",
        "konkuk": "건국대학교"
    }
    return mapping.get(eng_name.lower(), eng_name)
