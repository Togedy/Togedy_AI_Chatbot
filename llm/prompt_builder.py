from typing import List, Dict


def build_ner_prompt(question: str, ner_result: Dict[str, List[str]]) -> str:
    """
    사용자의 멀티 질문을 NER 결과를 바탕으로 구조화된 (UNI, TYPE, KEYWORD) 조합으로 재정렬하도록 GPT에게 요청하는 프롬프트 생성.
    """
    ner_text = f"""NER 라벨링 결과:
UNI: {ner_result.get("UNI", [])}
TYPE: {ner_result.get("TYPE", [])}
KEYWORD: {ner_result.get("KEYWORD", [])}

사용자 질문:
"{question}"

위 질문에서 사용자가 의미한 각 조합을 다음 형식으로 묶어서 JSON 배열로 만들어줘:

예시:
[
  {{
    "UNI": "서울대",
    "TYPE": "수시",
    "KEYWORD": "모집인원"
  }},
  {{
    "UNI": "연세대",
    "TYPE": "정시",
    "KEYWORD": "모집일정"
  }}
]

주의:
- 누락된 항목이 없도록 의미 있는 조합을 완성해줘.
- 다른 텍스트 없이 JSON만 출력해줘.
"""
    return ner_text


def build_answer_prompt(
    question_before: str,
    question_after: Dict[str, List[str]],
    documents: List[str]
) -> str:
    """
    GPT가 답변을 생성할 수 있도록 문서 정보와 질문 키워드를 포함한 프롬프트 생성
    """
    doc_text = "\n\n".join([f"[문서 {i+1}]\n{doc}" for i, doc in enumerate(documents)])

    prompt = f"""
[사용자 질문 원문]
{question_before}

[질문 키워드 정리 결과]
UNI: {question_after.get("UNI", [])}
TYPE: {question_after.get("TYPE", [])}
KEYWORD: {question_after.get("KEYWORD", [])}

[참고 문서]
{doc_text}

위 내용을 바탕으로 사용자의 질문에 대해 존댓말로 답변해줘.  
답변은 명확하고 간결하게 정리해줘.
"""
    return prompt
