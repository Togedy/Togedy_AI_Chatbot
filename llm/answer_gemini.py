# llm/answer_gemini.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import google.generativeai as genai

log = logging.getLogger("app.answer")

def _setup():
    load_dotenv(override=True)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY가 설정되어 있지 않습니다.")
    genai.configure(api_key=api_key)
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    return genai.GenerativeModel(model_name)

def generate_final_answer(
    question: str,
    validated: Dict[str, Any],
    page_hits: List[Dict[str, Any]],
    table_rows: Optional[List[Dict[str, Any]]] = None,
    lang: str = "ko"
) -> str:
    """
    page_hits 예:
    [
      {"source_text": ".../susi_text.txt", "uni_key":"konkuk", "type_key":"susi",
       "pages":[{"page":3,"score":0.42,"excerpt":"..."}]}
    ]
    table_rows: 표에서 고른 행들의 리스트(dict). 없으면 None.
    """
    model = _setup()

    # ---------- 컨텍스트 구성 ----------
    ctx_lines: List[str] = []
    ctx_lines.append(f"[질문] {question}")

    ents = validated.get("entities", {})
    def _vals(key): 
        return [e.get("normalized") or e.get("text") for e in ents.get(key, []) if (e.get("normalized") or e.get("text"))]

    ctx_lines.append(
        f"[엔티티] UNI={_vals('UNI')} TYPE={_vals('TYPE')} KEYWORD={_vals('KEYWORD')}"
    )

    for hit in page_hits:
        src = hit.get("source_text")
        uni_key, type_key = hit.get("uni_key"), hit.get("type_key")
        ctx_lines.append(f"\n[문서] uni={uni_key} type={type_key} text={src}")
        for p in hit.get("pages", []):
            page_no = p.get("page")
            score = p.get("score", 0.0)
            excerpt = (p.get("excerpt") or "").strip()
            ctx_lines.append(f"- Page {page_no} (score={score:.3f})\n  {excerpt}")

    if table_rows:
        ctx_lines.append("\n[표 추출 결과 상위 행]")
        # ⬇⬇⬇ 문제되던 줄: f-string 이스케이프 제거, 안전한 조합으로 변경
        for i, row in enumerate(table_rows[:5], 1):
            pairs = [f"{k}:{row[k]}" for k in row.keys()]
            ctx_lines.append("- row#{}: {}".format(i, " | ".join(pairs)))

    ctx = "\n".join(ctx_lines)

    # ---------- 프롬프트 ----------
    sys_prompt = (
        "당신은 한국 대입 정보 어시스턴트입니다. 주어진 컨텍스트(텍스트/표)만을 근거로 질문에 정확히 답하세요. "
        "없거나 불확실한 정보는 '자료에서 확인되지 않습니다'라고 명시하세요. "
        "답변은 간결한 한국어 문장으로, 필요한 경우 근거 문서(대학/전형, 페이지 번호)를 괄호로 덧붙이세요."
    )
    user_prompt = (
        f"{ctx}\n\n[요청]\n"
        "- 질문에 대한 최종 답변을 한글로 작성\n"
        "- 수치/모집인원은 단위와 출처(대학/전형, 페이지)를 괄호로 표기\n"
        "- 표/텍스트에 불일치가 있으면 가장 명확한 자료를 우선하고, 모순이 있음을 언급\n"
    )

    # ---------- 호출 ----------
    try:
        # 간단/안정한 형태로 전달
        resp = model.generate_content([sys_prompt, user_prompt])
        return (getattr(resp, "text", None) or "").strip()
    except Exception as e:
        log.warning("최종 답변 생성 실패: %s", e)
        return "가져온 문서와 표를 기준으로 답변을 생성하지 못했습니다. 문서와 표 내용을 확인해 주세요."
