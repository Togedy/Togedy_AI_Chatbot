# -*- coding: utf-8 -*-
import os
import json
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

SYSTEM_INSTRUCTION = """You are an admissions domain assistant.
Given a Korean question and extracted NER entities (UNI, TYPE, KEYWORD),
validate and, if necessary, correct them.

Return ONLY JSON that matches this spec (no other text):
{
  "verdict": "accept" | "corrected" | "reject",
  "reason": "string (Korean)",
  "entities": {
    "UNI": [{"text": "string","normalized": "string","source": "ner|llm|implied"}],
    "TYPE": [{"text": "string","normalized": "string","source": "ner|llm|implied"}],
    "KEYWORD": [{"text": "string","normalized": "string","source": "ner|llm|implied"}]
  }
}

Rules:
1) Do not invent facts beyond the question.
2) If strongly implied, you may add with source="implied".
3) normalized uses standard Korean forms (e.g., "서울대학교", "정시 일반전형").
4) Keep it concise and consistent.
"""

def _build_user_prompt(
    question: str,
    ner_entities: Dict[str, List[str]],
    allowed_universities: Optional[List[str]],
    allowed_types: Optional[List[str]],
) -> str:
    lines = []
    lines.append("사용자 질문:")
    lines.append(question.strip())
    lines.append("")
    lines.append("NER 추출 결과:")
    lines.append(f"- UNI: {', '.join(ner_entities.get('UNI', [])) or '(없음)'}")
    lines.append(f"- TYPE: {', '.join(ner_entities.get('TYPE', [])) or '(없음)'}")
    lines.append(f"- KEYWORD: {', '.join(ner_entities.get('KEYWORD', [])) or '(없음)'}")
    lines.append("")
    if allowed_universities:
        lines.append(f"[검증 기준] 허용 대학: {', '.join(allowed_universities)}")
    if allowed_types:
        lines.append(f"[검증 기준] 허용 전형: {', '.join(allowed_types)}")
    lines.append("")
    lines.append("요청:")
    lines.append("1) 위 엔티티가 질문과 의미상 일치하는지 검증/보정.")
    lines.append("2) 부족/모호 시 보정하고 source를 'ner'|'llm'|'implied'로 표기.")
    lines.append("3) 지정 JSON 스펙 외 텍스트 금지.")
    return "\n".join(lines)

def _configure_gemini(api_key: Optional[str] = None, model_name: Optional[str] = None):
    api_key = api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY가 설정되어 있지 않습니다 (.env 확인).")
    genai.configure(api_key=api_key)
    model = model_name or os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
    return genai.GenerativeModel(model)

def validate_with_gemini(
    question: str,
    ner_entities: Dict[str, List[str]],
    *,
    allowed_universities: Optional[List[str]] = None,
    allowed_types: Optional[List[str]] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    입력:
      - question: 사용자 질문(문장)
      - ner_entities: {"UNI":[...], "TYPE":[...], "KEYWORD":[...]}  ← NER 결과
    출력(JSON):
      {
        "verdict": "accept|corrected|reject",
        "reason": "...",
        "entities": {
          "UNI": [{"text","normalized","source"}],
          "TYPE": [...],
          "KEYWORD": [...]
        }
      }
    """
    # 기본 허용 목록(필요 시 외부에서 주입)
    if allowed_universities is None:
        allowed_universities = ["서울대", "연세대", "고려대", "서강대", "성균관대", "한양대", "건국대"]
    if allowed_types is None:
        allowed_types = ["수시", "정시", "논술전형", "학생부종합전형", "학생부교과전형", "특기자전형", "기타"]

    user_prompt = _build_user_prompt(
        question=question,
        ner_entities=ner_entities,
        allowed_universities=allowed_universities,
        allowed_types=allowed_types,
    )
    model = _configure_gemini(api_key, model_name)

    resp = model.generate_content(
        contents=[
            {"role": "user", "parts": [SYSTEM_INSTRUCTION]},
            {"role": "user", "parts": [user_prompt]},
        ],
        generation_config={
            "temperature": temperature,
            "response_mime_type": "application/json",  # JSON 강제
        },
    )

    # 응답 안전 처리
    if not hasattr(resp, "text") or not resp.text:
        raise RuntimeError("Gemini 응답이 비어 있습니다. 입력/키/모델을 확인하세요.")

    # JSON 파싱
    try:
        data = json.loads(resp.text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Gemini 응답 JSON 파싱 실패: {e}\n원문: {resp.text[:500]}")

    # 최소 스키마 보정
    data.setdefault("verdict", "accept")
    data.setdefault("reason", "")
    data.setdefault("entities", {})
    for k in ("UNI", "TYPE", "KEYWORD"):
        data["entities"].setdefault(k, [])
    return data

# 단독 실행 예시
if __name__ == "__main__":
    # 가정: 이미 NER이 뽑아준 결과
    question = "서울대 정시 일반전형에서 전기정보공학부 모집인원 알려줘."
    ner_entities = {
        "UNI": ["서울대"],
        "TYPE": ["정시", "일반전형"],
        "KEYWORD": ["모집인원"]
    }
    result = validate_with_gemini(question, ner_entities)
    print(json.dumps(result, ensure_ascii=False, indent=2))
