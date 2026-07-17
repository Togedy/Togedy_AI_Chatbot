# generate_answers.py
# -*- coding: utf-8 -*-

import os
import re
import time
import argparse
from collections import defaultdict
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from search_and_export import search_top_pages_for_query

from extract_all import (
    UniExtractor,
    TypeExtractor,
    KeywordExtractorBridge,
    load_env,
)

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


PAGE_LABEL_RE = re.compile(r"^\s*={2,}\s*Page\s*(\d+)\s*={2,}\s*$", re.IGNORECASE)

# =============================================================================
# Prompt templates
# =============================================================================

EXPERT_SYSTEM_PROMPT = (
    "당신은 한국 대학 입시(모집요강 기반) 질문에 답하는 챗봇입니다.\n"
    "말투는 사용자를 이해하고 함께 성장하는 친구처럼 친근하되, 항상 공손한 존댓말을 사용합니다.\n"
    "사용자가 불안해하거나 고민을 말해도 무시하지 말고, 핵심을 정리해서 차분히 안내합니다.\n\n"
    "답변 원칙:\n"
    "1) 모집요강 문서 컨텍스트/출처가 주어진 경우에는 반드시 그 내용만 근거로 답변합니다.\n"
    "2) 문서 컨텍스트가 없거나 부족한 경우에도, 일반적인 입시 흐름/경향을 바탕으로 도움이 되는 설명을 먼저 제공합니다.\n"
    "3) 문서가 없는 상태에서 수치/일정/규정/자격요건을 단정하지 않습니다.\n"
    "4) 답변 마지막에는 더 정확한 안내를 위해 필요한 '정확한 키워드' 1~2개를 공손하게 요청합니다.\n\n"
    "금지 규칙(출력에 포함하지 말 것):\n"
    "- '문서 근거 없음', '컨텍스트에서 확인 불가', '정확한 확인 필요' 같은 문구\n"
    "- 위와 유사한 형태의 직설적인 면책 문구\n\n"
    "형식:\n"
    "- 본문은 4~10문장 내외로 과도하게 길지 않게 작성합니다.\n"
    "- 마지막 줄에는 질문 형태로 키워드 요청 1문장을 붙입니다.\n"
)

DIRECT_ANSWER_USER_TEMPLATE = (
    "아래 질문에 대해 일반적인 입시 흐름/경향 또는 일반 상식을 바탕으로 답변해 주세요.\n"
    "사용자가 이미 연도, 대상, 키워드를 충분히 제공했다면 추가 질문을 하지 말고 바로 답변을 마무리하세요.\n"
    "정보가 부족할 때만 마지막에 필요한 정보를 1문장으로 질문하세요.\n\n"
    "출력 규칙:\n"
    "1) 설명: 4~8문장\n"
    "2) 사용자가 이미 충분한 정보를 제공했다면 추가 질문을 절대 하지 말 것\n"
    "3) 추가 질문이 필요한 경우에만 마지막에 1문장으로 질문할 것\n"
    "4) 이미 질문에 포함된 표현을 다시 물어보지 말 것\n"
    "5) 금지 문구는 절대 쓰지 말 것(문서/컨텍스트/정확 확인 관련 면책 문구)\n\n"
    "질문:\n"
    "{question}\n"
)

DOC_SEARCH_FAIL_USER_TEMPLATE = (
    "사용자가 특정 대학 또는 입시 정보를 물어봤지만, 현재 가지고 있는 모집요강 자료에서 충분히 답변할 근거를 찾기 어려운 상황입니다.\n"
    "사용자의 질문 의도를 바탕으로 일반적인 입시 안내를 먼저 제공하세요.\n"
    "단, 답변 마지막에는 반드시 사용자가 다음 질문에서 대학명(UNI)을 포함하도록 자연스럽게 유도하는 질문을 붙이세요.\n\n"
    "출력 규칙:\n"
    "1) 일반 설명: 3~6문장\n"
    "2) '문서', '컨텍스트', '검색 실패', '근거 없음', '확인 불가' 같은 표현은 직접적으로 쓰지 말 것\n"
    "3) 마지막 문장은 반드시 대학명(UNI)을 포함해서 다시 질문하도록 유도하는 질문이어야 함\n"
    "4) 마지막 질문에는 사용자의 질문 의도를 반영한 예시를 1개 포함할 것\n"
    "5) 예시 형식:\n"
    "   - 어느 대학 기준으로 확인해 드릴까요? 예: 건국대 수시 모집일정\n"
    "   - 대학명을 포함해서 다시 질문해 주시면 더 정확히 안내해 드릴게요. 예: 연세대 정시 제출서류 알려줘\n\n"
    "사용자 질문:\n"
    "{question}\n\n"
    "현재 추출된 정보:\n"
    "- UNI: {uni}\n"
    "- TYPE: {typ}\n"
    "- KEYWORD: {keywords}\n"
)

KEYWORD_ONLY_USER_TEMPLATE = (
    "사용자의 질문에서 대학(UNI) 또는 전형(TYPE)은 명확하지 않고, 키워드(KEYWORD)만 있는 상태입니다.\n"
    "따라서 먼저 키워드를 중심으로 일반적인 설명을 제공한 뒤, 더 정확한 안내를 위해 필요한 키워드를 요청하세요.\n\n"
    "출력 규칙:\n"
    "1) 키워드 중심 설명: 4~8문장\n"
    "2) 마지막: 더 정확한 안내를 위해 필요한 '정확한 키워드' 1~2개를 공손하게 질문 1문장으로 요청\n"
    "3) 금지 문구는 절대 쓰지 말 것(문서/컨텍스트/정확 확인 관련 면책 문구)\n\n"
    "추출된 키워드:\n"
    "{keywords}\n\n"
    "사용자 질문:\n"
    "{question}\n"
)

DOC_ANSWER_USER_TEMPLATE = (
    "아래는 모집요강 문서 발췌(컨텍스트)와 출처입니다.\n"
    "반드시 컨텍스트에 있는 내용만 근거로 답하세요.\n"
    "컨텍스트에 없는 정보는 추측하지 말고, 대신 사용자가 제공하면 좋은 키워드를 마지막에 1문장으로 요청하세요.\n\n"
    "질문:\n"
    "{question}\n\n"
    "컨텍스트:\n"
    "{context}\n\n"
    "출처:\n"
    "{sources}\n\n"
    "출력 규칙:\n"
    "1) 핵심 결론을 먼저 1~3문장으로 제시\n"
    "2) 표/수치가 나오면, 컨텍스트에서 근거가 되는 항목을 짧게 언급\n"
    "3) 마지막: 더 정확한 안내를 위해 필요한 '정확한 키워드' 1~2개를 공손하게 질문 1문장으로 요청\n"
    "4) 금지 문구는 절대 쓰지 말 것(문서/컨텍스트/정확 확인 관련 면책 문구)\n"
)

FOLLOWUP_QUESTION_PROMPT_TEMPLATE = (
    "당신의 목표는 사용자가 다음 턴에 바로 답할 수 있는 '재질문' 1문장만 만드는 것입니다.\n\n"
    "규칙:\n"
    "1) 반드시 한 문장 질문으로만 출력합니다.\n"
    "2) 한 번에 1~2개의 정보만 요청합니다.\n"
    "3) 공손한 존댓말로 질문합니다.\n"
    "4) 다음 표현은 절대 쓰지 말 것: '문서 근거 없음', '컨텍스트에서 확인 불가', '정확한 확인 필요'.\n"
    "5) 아래 NER 결과를 참고해, 가장 부족한 핵심정보를 우선으로 묻습니다.\n\n"
    "우선순위:\n"
    "- 대학(UNI)이 없으면: 대학명을 먼저 묻기\n"
    "- 전형(TYPE)이 없으면: 전형명을 먼저 묻기\n"
    "- 학과/모집단위(KEYWORD)가 없으면: 학과/모집단위를 묻기\n"
    "- 일정/모집 관련이면 연도(예: 2026학년도)를 묻기\n\n"
    "사용자 원문 질문:\n{question}\n\n"
    "현재 NER 결과:\n"
    "- UNI: {uni}\n"
    "- TYPE: {typ}\n"
    "- KEYWORD: {kw}\n\n"
    "출력: 재질문 한 문장"
)


def call_llm(
    user_prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.1,
    system_prompt: str = EXPERT_SYSTEM_PROMPT,
) -> str:
    if OpenAI is None:
        return "현재 LLM 호출 환경이 준비되지 않아 답변을 생성하기 어렵습니다. 실행 환경을 먼저 확인해 주실 수 있을까요?"
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return "현재 LLM 호출 키가 설정되어 있지 않아 답변을 생성하기 어렵습니다. 실행 환경을 먼저 확인해 주실 수 있을까요?"

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


def gpt_chat(
    system_prompt: str,
    user_prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.1,
) -> str:
    return call_llm(
        user_prompt=user_prompt,
        model=model,
        temperature=temperature,
        system_prompt=system_prompt,
    )


def clean_rewritten_question(text: str) -> str:
    """
    LLM이 재작성 결과를 코드블록/라벨 형태로 반환했을 때 정리한다.
    """
    text = (text or "").strip()

    text = re.sub(r"^```(?:text|json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    prefixes = [
        "재작성된 질문:",
        "최종 질문:",
        "질문:",
        "resolved_question:",
        "출력:",
    ]

    for p in prefixes:
        if text.startswith(p):
            text = text[len(p):].strip()

    return text.strip('"').strip("'").strip()


def _format_prev_ner_for_prompt(prev_ner: Optional[Dict[str, Any]]) -> str:
    """
    프론트가 넘긴 NER 또는 서버 내부 NER를 후속 질문 재작성 프롬프트에 넣기 좋게 정리한다.
    """
    prev_ner = prev_ner or {}

    def pick(*keys):
        for k in keys:
            v = prev_ner.get(k)
            if v:
                return v
        return []

    def to_list(v):
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        s = str(v).strip()
        return [s] if s else []

    uni = to_list(pick("UNI", "uni"))
    typ = to_list(pick("TYPE", "type"))
    kw = to_list(pick("KEYWORD", "keywords"))

    return (
        f"- UNI: {', '.join(uni) if uni else '(없음)'}\n"
        f"- TYPE: {', '.join(typ) if typ else '(없음)'}\n"
        f"- KEYWORD: {', '.join(kw) if kw else '(없음)'}"
    )


def rewrite_followup_question(
    bot_question: str,
    user_input: str,
    prev_ner: Optional[Dict[str, Any]] = None,
    model: str = "gpt-4o-mini",
) -> str:
    """
    first=false일 때 사용한다.

    입력 의미:
    - bot_question: 직전 챗봇 질문(question_1)
    - user_input: 사용자의 두 번째 입력(question_2)
    - prev_ner: 프론트가 유지하고 있던 이전 NER 값

    출력:
    - 기존 NER/검색 파이프라인에 넣을 수 있는 최종 질문 1문장
    - 사용자가 명확히 거절하면 "추가 질문 없음"
    """
    bot_question = (bot_question or "").strip()
    user_input = (user_input or "").strip()

    if not user_input:
        return ""

    prev_ner_text = _format_prev_ner_for_prompt(prev_ner)

    system_prompt = """너는 한국 대학 입시 챗봇의 후속 입력 재작성기다.
직전 챗봇 질문과 사용자 입력을 보고, 서버가 실제로 답변해야 할 질문을 한국어 한 문장으로 재작성한다.
설명, JSON, 코드블록 없이 최종 질문 한 문장만 출력한다."""

    user_prompt = f"""
[직전 챗봇 질문]
{bot_question}

[이전 턴 NER 정보]
{prev_ner_text}

[사용자 입력]
{user_input}

재작성 규칙:
1. 사용자가 "어", "응", "네", "예", "알려줘", "좋아"처럼 답하면 직전 챗봇 질문에서 물어본 정보를 요청한 것으로 재작성한다.
2. 사용자가 직전 챗봇 질문에 조건을 추가하면, 직전 챗봇 질문과 사용자 입력을 합쳐서 재작성한다.
3. 사용자가 "그럼", "그러면", "정시는?", "수시는?", "제출서류는?", "일정은?"처럼 이전 맥락에 의존하는 질문을 하면 이전 턴 NER 정보를 반영해 재작성한다.
4. 사용자가 완전히 다른 주제를 말하면, 사용자 입력을 새 질문으로 재작성한다.
5. 사용자가 "아니", "아니요", "필요 없어", "괜찮아"처럼 부정만 말하면 "추가 질문 없음"이라고만 출력한다.
6. 단, "아니 연세대", "아니 정시", "아니 제출서류", "아니 건국대 컴퓨터공학부"처럼 부정 뒤에 새로운 정보가 있으면 거절이 아니라 기존 질문을 수정한 것으로 판단해 재작성한다.
7. 출력은 반드시 최종 질문 한 문장만 한다.

예시:
직전 챗봇 질문: 수능 날짜에 대한 정보가 필요하신가요?
사용자 입력: 어
출력: 수능 날짜에 대한 정보를 알려줘

직전 챗봇 질문: 수능 날짜에 대한 정보가 필요하신가요?
사용자 입력: 2026년 기준으로 알려줘
출력: 2026년 수능 날짜에 대한 정보를 알려줘

직전 챗봇 질문: 수능 날짜에 대한 정보가 필요하신가요?
사용자 입력: 나는 건국대 컴퓨터공학부 입시 정보가 궁금해
출력: 건국대 컴퓨터공학부 입시 정보를 알려줘

직전 챗봇 질문: 어느 대학의 정시 전형방법을 알려드릴까요?
이전 턴 NER 정보:
- UNI: 건국대
- TYPE: 정시
- KEYWORD: 전형방법
사용자 입력: 제출서류도 알려줘
출력: 건국대 정시 제출서류를 알려줘

직전 챗봇 질문: 건국대 수시 전형방법이 궁금하신가요?
이전 턴 NER 정보:
- UNI: 건국대
- TYPE: 수시
- KEYWORD: 전형방법
사용자 입력: 아니 연세대
출력: 연세대 수시 전형방법을 알려줘
"""

    raw = gpt_chat(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        temperature=0.0,
    )

    rewritten = clean_rewritten_question(raw)

    if not rewritten:
        return user_input

    return rewritten


def build_followup_prompt(question: str, ner_uni, ner_type, ner_keywords) -> str:
    def to_list(x):
        if x is None:
            return []
        if isinstance(x, list):
            return x
        return [x]

    u = [str(v).strip() for v in to_list(ner_uni) if str(v).strip()]
    t = [str(v).strip() for v in to_list(ner_type) if str(v).strip()]
    k = [str(v).strip() for v in to_list(ner_keywords) if str(v).strip()]

    uni_str = ", ".join(u) if u else "(없음)"
    typ_str = ", ".join(t) if t else "(없음)"
    kw_str = ", ".join(k) if k else "(없음)"

    return FOLLOWUP_QUESTION_PROMPT_TEMPLATE.format(
        question=question,
        uni=uni_str,
        typ=typ_str,
        kw=kw_str,
    )


def is_quota_question(text: str) -> bool:
    if not text:
        return False
    t = text.replace(" ", "")
    triggers = [
        "몇명", "몇명을", "몇명뽑", "몇명뽑는", "몇명뽑는지", "몇명뽑아", "몇명뽑아요",
        "모집인원", "모집인원은", "모집인원알려", "모집인원알려줘",
        "선발", "선발인원", "선발인원알려", "선발인원알려줘",
        "정원", "정원은", "정원내", "정원외",
        "인원", "인원수",
    ]
    return any(k in t for k in triggers)


def split_pages_with_label(raw: str) -> List[Dict[str, Any]]:
    pages: List[Dict[str, Any]] = []
    cur_label: Optional[int] = None
    buf: List[str] = []

    for line in (raw or "").splitlines():
        m = PAGE_LABEL_RE.match(line.strip())
        if m:
            if buf:
                text = "\n".join(buf).strip()
                if text:
                    pages.append({"label": cur_label, "text": text})
                buf = []
            cur_label = int(m.group(1))
            continue
        buf.append(line)

    if buf:
        text = "\n".join(buf).strip()
        if text:
            pages.append({"label": cur_label, "text": text})

    return pages


@lru_cache(maxsize=128)
def load_doc_text(doc_path: str) -> str:
    with open(doc_path, "r", encoding="utf-8") as f:
        return f.read()


def load_page_text(doc_path: str, page_label: int) -> str:
    raw = load_doc_text(doc_path)
    page_objs = split_pages_with_label(raw)
    for p in page_objs:
        if p["label"] == page_label:
            return p["text"]
    return ""


def pick_context_from_rows(rows: List[Dict[str, Any]], topk: int = 3) -> str:
    if not rows:
        return ""

    valid = [r for r in rows if r.get("page_index", -1) != -1]
    if not valid:
        valid = rows[:]

    valid = sorted(valid, key=lambda r: float(r.get("score", 0.0)), reverse=True)

    blocks: List[str] = []
    for r in valid[: max(1, int(topk))]:
        doc_path = str(r.get("doc_path", "") or "")
        page = r.get("page_index", -1)

        body = r.get("text") or r.get("snippet") or r.get("content") or ""
        body = str(body).strip()

        meta_parts = []
        if doc_path:
            meta_parts.append(os.path.basename(doc_path))
        if isinstance(page, int) and page != -1:
            meta_parts.append(f"p.{page}")

        meta = " | ".join(meta_parts).strip()
        if meta:
            blocks.append(f"[{meta}]\n{body}".strip())
        else:
            blocks.append(body)

    return "\n\n".join([b for b in blocks if b]).strip()


def build_sources_from_rows(rows: List[Dict[str, Any]], topk: int = 3) -> str:
    if not rows:
        return ""

    valid = [r for r in rows if r.get("page_index", -1) != -1]
    if not valid:
        valid = rows[:]

    valid = sorted(valid, key=lambda r: float(r.get("score", 0.0)), reverse=True)

    seen = set()
    lines: List[str] = []

    for r in valid[: max(1, int(topk))]:
        doc_path = str(r.get("doc_path", "") or "")
        page = r.get("page_index", -1)

        doc_name = os.path.basename(doc_path) if doc_path else "(unknown)"
        page_str = f"p.{page}" if isinstance(page, int) and page != -1 else "(page unknown)"

        key = (doc_name, page_str)
        if key in seen:
            continue
        seen.add(key)

        matched_uni = (r.get("matched_uni") or "").strip()
        matched_type = (r.get("matched_type") or "").strip()

        if matched_uni or matched_type:
            ut = " / ".join([x for x in [matched_uni, matched_type] if x])
            lines.append(f"- {ut}: {doc_name} {page_str}")
        else:
            lines.append(f"- {doc_name} {page_str}")

    return "\n".join(lines).strip()


def build_sources(pair_to_rows: Dict[Tuple[str, str], List[Dict[str, Any]]]) -> List[str]:
    out: List[str] = []
    for (uni, typ), rows in pair_to_rows.items():
        pages: List[int] = []
        for r in rows:
            try:
                pages.append(int(r.get("page_index")))
            except Exception:
                pass
        pages = sorted(set(pages))
        if pages:
            pages_str = ", ".join([f"p.{p}" for p in pages])
            out.append(f"- {uni} , {typ} 모집요강 {pages_str}")
        else:
            out.append(f"- {uni} , {typ} 모집요강 (페이지 정보 없음)")
    return out


def build_quota_sources(selected_pages: Dict[Tuple[str, str], List[int]]) -> List[str]:
    out: List[str] = []
    for (uni, typ), pages in selected_pages.items():
        pages_sorted = sorted(set([int(p) for p in pages if isinstance(p, int) or str(p).isdigit()]))
        if pages_sorted:
            pages_str = ", ".join([f"p.{p}" for p in pages_sorted])
            out.append(f"- {uni} , {typ} 모집요강 {pages_str}")
    return out


def pick_best_quota_pages(rows: List[Dict[str, Any]], majors: List[str], max_pages: int = 2) -> List[Dict[str, Any]]:
    if not rows:
        return []

    majors = [m for m in (majors or []) if m and str(m).strip()]

    def contains_major(snippet: str) -> bool:
        if not majors:
            return True
        s = snippet or ""
        return any(m in s for m in majors)

    rows_sorted = sorted(rows, key=lambda r: float(r.get("score", 0.0)), reverse=True)

    picked: List[Dict[str, Any]] = []
    used_pages = set()

    for r in rows_sorted:
        try:
            pno = int(r.get("page_index"))
        except Exception:
            continue
        if pno in used_pages:
            continue
        if contains_major(r.get("snippet", "")):
            picked.append(r)
            used_pages.add(pno)
        if len(picked) >= max_pages:
            return picked

    for r in rows_sorted:
        try:
            pno = int(r.get("page_index"))
        except Exception:
            continue
        if pno in used_pages:
            continue
        picked.append(r)
        used_pages.add(pno)
        if len(picked) >= max_pages:
            break

    return picked


def build_quota_prompt(
    question: str,
    uni: str,
    typ: str,
    majors: List[str],
    page_texts: List[Tuple[int, str]],
) -> str:
    majors_str = ", ".join([m for m in majors if m]) if majors else "(학과/학부명 미추출)"
    pages_info = ", ".join([f"p.{pno}" for pno, _ in page_texts if isinstance(pno, int)])

    doc_block = []
    for pno, txt in page_texts:
        doc_block.append(f"[p.{pno}]\n{txt}")
    doc_join = "\n\n".join(doc_block)

    return (
        "아래는 대학 모집요강 텍스트(페이지 발췌)입니다.\n"
        "반드시 제공된 텍스트 안의 정보만 근거로 답변해 주세요.\n\n"
        f"질문: {question}\n"
        f"대학/전형: {uni} / {typ}\n"
        f"대상 학과/학부: {majors_str}\n"
        f"참고 페이지: {pages_info}\n\n"
        "요구사항:\n"
        "1) 대상 학과/학부의 모집인원 합계를 먼저 제시해 주세요.\n"
        "2) 전형별 항목(예: 지역균형/일반전형/기회균형특별전형 등)이 있으면 함께 정리해 주세요.\n"
        "3) 답변은 한두 문장으로 간단명료하게 작성해 주세요.\n"
        "4) 추가 질문이나 안내 문구는 절대 덧붙이지 말고, 모집인원 답변만 작성해 주세요.\n\n"
        "제공 문서:\n"
        "-----\n"
        f"{doc_join}\n"
        "-----\n"
    )


def sanitize_pair_answer(text: str) -> str:
    if not text:
        return ""

    cleaned = text.strip()

    split_patterns = [
        r"\n\s*혹시",
        r"\n\s*추가로",
        r"\n\s*더 궁금",
        r"\n\s*궁금한 점",
        r"\n\s*필요한 키워드",
    ]
    for pat in split_patterns:
        cleaned = re.split(pat, cleaned, maxsplit=1)[0].strip()

    cleaned = cleaned.split("\n\n")[0].strip()
    return cleaned


def build_pair_doc_context(prows: List[Dict[str, Any]], max_items: int = 3) -> str:
    if not prows:
        return ""

    rows_sorted = sorted(prows, key=lambda r: float(r.get("score", 0.0)), reverse=True)
    lines: List[str] = []

    for r in rows_sorted[:max(1, int(max_items))]:
        page_index = r.get("page_index")
        snippet = str(r.get("snippet", "") or "").strip()
        if isinstance(page_index, int):
            lines.append(f"[p.{page_index}] {snippet}")
        else:
            lines.append(snippet)

    return "\n".join([x for x in lines if x]).strip()


def answer_one(
    text: str,
    uni_ex: UniExtractor,
    type_ex: TypeExtractor,
    kw_ex: KeywordExtractorBridge,
    api_key: str,
    gemini_model: str,
    llm_model: str = "gpt-4o-mini",
    top_pages: int = 3,
    quota_pages_per_pair: int = 2,
) -> Dict[str, Any]:
    rows, stats, ner = search_top_pages_for_query(
        text,
        uni_ex,
        type_ex,
        kw_ex,
        api_key,
        gemini_model,
        top_pages=top_pages,
    )

    decision = (ner.get("decision", "") or "").strip()

    ner_uni = ner.get("uni") or []
    ner_type = ner.get("type") or []
    ner_kw = ner.get("keywords") or []
    if not isinstance(ner_uni, list):
        ner_uni = [ner_uni]
    if not isinstance(ner_type, list):
        ner_type = [ner_type]
    if not isinstance(ner_kw, list):
        ner_kw = [ner_kw]

    pair_to_rows: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        uni = (r.get("matched_uni") or "").strip()
        typ = (r.get("matched_type") or "").strip()
        if not uni or not typ:
            continue
        pair_to_rows[(uni, typ)].append(r)

    has_uni = len([x for x in ner_uni if str(x).strip()]) > 0
    has_type = len([x for x in ner_type if str(x).strip()]) > 0
    has_kw = len([x for x in ner_kw if str(x).strip()]) > 0

    if (not has_uni) and (not has_type) and has_kw:
        decision = "키워드답변"

    answer_text = ""
    sources_lines: List[str] = []

    if decision == "키워드답변":
        keywords_str = ", ".join([str(k).strip() for k in ner_kw if str(k).strip()]) or "(미추출)"
        user_prompt = KEYWORD_ONLY_USER_TEMPLATE.format(
            question=text,
            keywords=keywords_str,
        )

        main_answer = gpt_chat(
            system_prompt=EXPERT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=llm_model,
            temperature=0.3,
        ).strip()

        # 재질문은 run_single_turn/run_followup_turn에서 부족 슬롯을 기준으로
        # 한 번만 붙인다. 여기서는 본문만 반환해 중복 질문을 방지한다.
        answer_text = main_answer.strip()
        sources_lines = []

        return {
            "input": text,
            "ner_uni": ner_uni,
            "ner_type": ner_type,
            "ner_kw": ner_kw,
            "decision": decision,
            "stats": stats,
            "pair_to_rows": pair_to_rows,
            "answer": answer_text,
            "sources": sources_lines,
        }

    if decision != "문서탐색":
        user_prompt = DIRECT_ANSWER_USER_TEMPLATE.format(question=text)

        main_answer = gpt_chat(
            system_prompt=EXPERT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=llm_model,
            temperature=0.3,
        ).strip()

        # 재질문은 run_single_turn/run_followup_turn에서 부족 슬롯을 기준으로
        # 한 번만 붙인다. 여기서는 본문만 반환해 중복 질문을 방지한다.
        answer_text = main_answer.strip()
        sources_lines = []

        return {
            "input": text,
            "ner_uni": ner_uni,
            "ner_type": ner_type,
            "ner_kw": ner_kw,
            "decision": decision,
            "stats": stats,
            "pair_to_rows": pair_to_rows,
            "answer": answer_text,
            "sources": sources_lines,
        }

    if is_quota_question(text):
        majors = [k for k in ner_kw if k and str(k).strip()]
        lines = ["모집 인원은 다음과 같습니다.\n"]
        selected_pages: Dict[Tuple[str, str], List[int]] = {}

        for (uni, typ), prows in pair_to_rows.items():
            picks = pick_best_quota_pages(prows, majors, max_pages=quota_pages_per_pair)
            if not picks:
                lines.append(f"- {uni} {typ}: 해당 항목을 정확히 보려면 학과나 모집단위 키워드를 조금 더 알려주세요.")
                selected_pages[(uni, typ)] = []
                continue

            page_texts: List[Tuple[int, str]] = []
            used_page_nums: List[int] = []

            for p in picks:
                doc_path = p.get("doc_path", "")
                try:
                    page_no = int(p.get("page_index"))
                except Exception:
                    continue
                if not doc_path or not os.path.exists(doc_path):
                    continue
                page_txt = load_page_text(doc_path, page_no)
                if not page_txt.strip():
                    continue
                page_texts.append((page_no, page_txt))
                used_page_nums.append(page_no)

            selected_pages[(uni, typ)] = used_page_nums

            if not page_texts:
                lines.append(f"- {uni} {typ}: 모집단위(학과/학부) 키워드를 조금 더 알려주시면 더 정확히 안내해 드릴 수 있습니다.")
                continue

            prompt = build_quota_prompt(text, uni, typ, majors, page_texts)
            quota_ans = call_llm(
                user_prompt=prompt,
                model=llm_model,
                temperature=0.0,
                system_prompt=EXPERT_SYSTEM_PROMPT,
            ).strip()

            quota_ans = sanitize_pair_answer(quota_ans)
            lines.append(f"- {uni} {typ}: {quota_ans}")

        answer_text = "\n".join(lines).strip()

        if not majors:
            answer_text += "\n\n어느 학과나 모집단위를 기준으로 보시는지 알려주시면 더 정확하게 안내해 드릴 수 있습니다."

        sources_lines = build_quota_sources(selected_pages)

        return {
            "input": text,
            "ner_uni": ner_uni,
            "ner_type": ner_type,
            "ner_kw": ner_kw,
            "decision": decision,
            "stats": stats,
            "pair_to_rows": pair_to_rows,
            "answer": answer_text,
            "sources": sources_lines,
        }

    # -------------------------------------------------------------------------
    # 문서탐색 답변: 대학/전형 페어별로 각각 GPT 호출
    # -------------------------------------------------------------------------
    lines: List[str] = []

    for (uni, typ), prows in pair_to_rows.items():
        pair_context = build_pair_doc_context(prows, max_items=top_pages)
        pair_sources = build_sources_from_rows(prows, topk=top_pages)

        if not pair_context.strip():
            continue

        doc_user_prompt = DOC_ANSWER_USER_TEMPLATE.format(
            question=f"{text}\n\n단, 이번 답변은 반드시 {uni} {typ}에 대해서만 답변해 주세요.",
            context=pair_context,
            sources=pair_sources,
        )

        pair_answer = gpt_chat(
            system_prompt=EXPERT_SYSTEM_PROMPT,
            user_prompt=doc_user_prompt,
            model=llm_model,
            temperature=0.1,
        ).strip()

        pair_answer = sanitize_pair_answer(pair_answer)

        if pair_answer:
            lines.append(f"{uni} {typ}\n{pair_answer}")

    # pair 결과가 하나도 없을 때만 기존 전체 컨텍스트 방식으로 fallback
    if lines:
        answer_text = "\n\n".join(lines).strip()
    else:
        context_blocks: List[str] = []
        for (uni, typ), prows in pair_to_rows.items():
            prows_sorted = sorted(prows, key=lambda r: int(r.get("page_index", 10**9)))
            snippet_join = "\n".join([f"[p.{r.get('page_index')}] {r.get('snippet','')}" for r in prows_sorted])
            context_blocks.append(f"({uni}, {typ}) 컨텍스트:\n{snippet_join}")

        sources_str = "\n".join(build_sources(pair_to_rows))

        doc_user_prompt = DOC_ANSWER_USER_TEMPLATE.format(
            question=text,
            context="\n\n".join(context_blocks).strip(),
            sources=sources_str.strip(),
        )

        answer_text = gpt_chat(
            system_prompt=EXPERT_SYSTEM_PROMPT,
            user_prompt=doc_user_prompt,
            model=llm_model,
            temperature=0.1,
        ).strip()

    sources_lines = build_sources(pair_to_rows)
    sources_lines = [s for s in sources_lines if "(페이지 정보 없음)" not in s]

    return {
        "input": text,
        "ner_uni": ner_uni,
        "ner_type": ner_type,
        "ner_kw": ner_kw,
        "decision": decision,
        "stats": stats,
        "pair_to_rows": pair_to_rows,
        "answer": answer_text,
        "sources": sources_lines,
    }



# =============================================================================
# 서버 연동용 대화 문맥 처리
# =============================================================================

ADMISSION_TERMS = [
    "입시", "대학", "대학교", "수시", "정시", "전형", "학과", "학부",
    "모집", "지원", "합격", "경쟁률", "등급", "내신", "수능", "논술",
    "학생부", "교과", "종합", "면접", "자소서", "자기소개서", "원서",
    "등록", "추가합격", "충원", "모집요강", "전공", "입학", "편입",
]

PROMPT_INJECTION_TERMS = [
    "앞선 모든", "앞의 모든", "이전 지시", "기존 지시", "모든 설명",
    "프롬프트를 무시", "규칙을 무시", "지시를 무시", "역할을 무시",
    "시스템 프롬프트", "명령을 무시",
]

OUT_OF_DOMAIN_TERMS = [
    "데이트", "맛집", "여행", "날씨", "주식", "코인", "연애", "영화",
    "게임", "요리", "식당", "배고파", "배고프", "점심", "저녁메뉴",
]

NEGATIVE_ONLY_PATTERNS = [
    r"^\s*(아니|아니요|됐어|괜찮아|필요\s*없어|필요없어|그만)\s*[.!?]*\s*$"
]


def _slot_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        values = value
    else:
        values = [value]

    out: List[str] = []
    for item in values:
        s = str(item).strip()
        if s and s not in out:
            out.append(s)
    return out


def normalize_ner(ner: Optional[Dict[str, Any]]) -> Dict[str, List[str]]:
    ner = ner or {}
    return {
        "UNI": _slot_list(ner.get("UNI", ner.get("uni", ner.get("UNT")))),
        "TYPE": _slot_list(ner.get("TYPE", ner.get("type"))),
        "KEYWORD": _slot_list(ner.get("KEYWORD", ner.get("keywords"))),
    }


# 짧은 후속 질문에서 모델이 놓치기 쉬운 명시적 표현을 보정한다.
# 현재 질문에 직접 적힌 값은 이전 NER보다 우선되어야 하므로,
# 모델 결과를 만든 직후 이 규칙을 적용한다.
EXPLICIT_KEYWORD_RULES = [
    ("제출서류", ["제출서류", "제출 서류", "서류도", "서류는", "필요서류", "필요 서류"]),
    ("전형방법", ["전형방법", "전형 방법", "평가방법", "평가 방법", "평가방식", "평가 방식"]),
    ("모집인원", ["모집인원", "모집 인원", "선발인원", "선발 인원", "몇명", "몇 명", "정원"]),
    ("모집일정", ["모집일정", "모집 일정", "원서접수", "원서 접수", "접수일정", "접수 일정", "일정은", "일정도"]),
    ("지원자격", ["지원자격", "지원 자격", "자격요건", "자격 요건"]),
    ("경쟁률", ["경쟁률"]),
    ("합격자발표", ["합격자발표", "합격자 발표", "합격발표", "합격 발표"]),
    ("등록기간", ["등록기간", "등록 기간", "등록일정", "등록 일정"]),
    ("추가합격", ["추가합격", "추가 합격", "충원합격", "충원 합격"]),
]


def apply_explicit_followup_rules(
    text: str,
    ner: Optional[Dict[str, Any]],
) -> Dict[str, List[str]]:
    """
    '정시는?', '제출서류도 알려줘'처럼 짧고 명확한 후속 입력을 보정한다.

    규칙으로 발견된 TYPE/KEYWORD는 사용자가 현재 문장에 직접 적은 값이므로
    모델 결과를 덮어쓴다. 대학명은 기존 UNI 모델 결과를 그대로 사용한다.
    """
    result = normalize_ner(ner)
    raw = (text or "").strip()
    compact = re.sub(r"\s+", "", raw).lower()

    # 전형명은 짧은 조사 결합 표현도 확실하게 인식한다.
    if "정시" in compact:
        result["TYPE"] = ["정시"]
    elif "수시" in compact:
        result["TYPE"] = ["수시"]

    for canonical, variants in EXPLICIT_KEYWORD_RULES:
        if any(re.sub(r"\s+", "", v).lower() in compact for v in variants):
            result["KEYWORD"] = [canonical]
            break

    return result


def _extract_ner_direct(
    text: str,
    uni_ex: UniExtractor,
    type_ex: TypeExtractor,
    kw_ex: KeywordExtractorBridge,
) -> Dict[str, List[str]]:
    try:
        uni = _slot_list(uni_ex.extract_uni(text))
    except Exception:
        uni = []

    try:
        typ = _slot_list(type_ex.extract_type(text))
    except Exception:
        typ = []

    try:
        keywords = _slot_list(kw_ex.extract_keywords(text))
    except Exception:
        keywords = []

    extracted = {
        "UNI": uni,
        "TYPE": typ,
        "KEYWORD": keywords,
    }

    return apply_explicit_followup_rules(text, extracted)


def _contains_any(text: str, terms: List[str]) -> bool:
    compact = (text or "").replace(" ", "").lower()
    return any(term.replace(" ", "").lower() in compact for term in terms)


def is_negative_only(text: str) -> bool:
    return any(re.match(pattern, text or "", re.IGNORECASE) for pattern in NEGATIVE_ONLY_PATTERNS)


def is_admission_related(text: str, ner: Optional[Dict[str, Any]] = None) -> bool:
    """
    현재 사용자 입력 자체가 입시 질문인지 판단한다.

    이전 NER이 존재한다는 이유만으로 데이트·맛집 같은 새 질문을 입시 질문으로
    강제 변환하지 않도록, 현재 문장의 표현과 현재 문장에서 추출된 NER만 사용한다.
    """
    text = (text or "").strip()
    cur = normalize_ner(ner)

    if cur["UNI"] or cur["TYPE"]:
        return True

    if _contains_any(text, ADMISSION_TERMS):
        return True

    # 학과명처럼 KEYWORD만 추출된 경우도 입시 문맥으로 본다.
    if cur["KEYWORD"] and not _contains_any(text, OUT_OF_DOMAIN_TERMS):
        return True

    return False


def is_out_of_domain_or_injection(text: str, ner: Optional[Dict[str, Any]] = None) -> bool:
    text = (text or "").strip()

    has_injection = _contains_any(text, PROMPT_INJECTION_TERMS)
    has_out_domain = _contains_any(text, OUT_OF_DOMAIN_TERMS)
    admission = is_admission_related(text, ner)

    # 입시 질문이 함께 들어 있으면 입시 부분은 처리할 수 있으므로 차단하지 않는다.
    return (has_injection or has_out_domain) and not admission


def build_domain_redirect_answer(text: str) -> str:
    text = (text or "").strip()

    if "배고" in text or any(k in text for k in ["점심", "저녁메뉴", "식사"]):
        return (
            "배가 고프시군요. 우선 식사를 잘 챙겨 드세요!\n"
            "저는 대학 입시 정보를 안내하는 챗봇입니다. "
            "궁금한 대학명, 학과, 수시·정시전형 또는 모집일정을 말씀해 주세요. "
            "예: 건국대 경영학과 수시전형 알려줘"
        )

    return (
        "저는 대학 입시 정보를 안내하는 챗봇이므로 요청하신 일반 주제 대신 "
        "대학별 수시·정시전형, 학과, 모집요강, 입시 일정에 대해 안내해 드릴 수 있습니다. "
        "예: 건국대 경영학과 수시전형 알려줘"
    )


def _is_uni_like_keyword(keyword: str, unis: List[str]) -> bool:
    k = (keyword or "").replace(" ", "")
    if not k:
        return False

    for uni in unis:
        u = (uni or "").replace(" ", "")
        if not u:
            continue
        if k == u or k in u or u in k:
            return True

    return k.endswith("대") or k.endswith("대학교")


def merge_ner_context(
    previous: Optional[Dict[str, Any]],
    current: Optional[Dict[str, Any]],
) -> Dict[str, List[str]]:
    """
    현재 질문에서 추출된 값이 있으면 현재 값을 우선하고,
    현재 질문에 없는 슬롯만 이전 NER에서 이어받는다.

    예:
    - 이전: UNI=건국대
      현재: TYPE=수시, KEYWORD=경영학과
      결과: 건국대 + 수시 + 경영학과

    - 이전: TYPE=수시, KEYWORD=경영학과
      현재: UNI=건국대
      결과: 건국대 + 수시 + 경영학과
    """
    prev = normalize_ner(previous)
    cur = normalize_ner(current)

    merged = {
        "UNI": cur["UNI"] if cur["UNI"] else prev["UNI"],
        "TYPE": cur["TYPE"] if cur["TYPE"] else prev["TYPE"],
        "KEYWORD": cur["KEYWORD"] if cur["KEYWORD"] else prev["KEYWORD"],
    }

    # UNI가 KEYWORD에도 중복 추출된 경우 제거한다.
    merged["KEYWORD"] = [
        kw for kw in merged["KEYWORD"]
        if not _is_uni_like_keyword(kw, merged["UNI"])
    ]

    return merged


def build_resolved_question(
    user_input: str,
    merged_ner: Optional[Dict[str, Any]],
) -> str:
    """
    병합된 NER를 검색 파이프라인에 안정적으로 전달할 수 있는 한 문장으로 만든다.
    """
    ner = normalize_ner(merged_ner)
    parts: List[str] = []

    parts.extend(ner["UNI"])
    parts.extend(ner["TYPE"])
    parts.extend(ner["KEYWORD"])

    # 원문에 NER로 잡히지 않은 일정, 서류, 경쟁률 등의 의도가 있을 수 있어 보존한다.
    raw = (user_input or "").strip()
    if raw and raw not in parts:
        parts.append(raw)

    deduped: List[str] = []
    for part in parts:
        p = str(part).strip()
        if p and p not in deduped:
            deduped.append(p)

    return " ".join(deduped).strip()


def missing_slots_for_followup(ner: Optional[Dict[str, Any]]) -> List[str]:
    """
    문서 탐색형 입시 답변에 필요한 핵심 슬롯을 반환한다.
    """
    n = normalize_ner(ner)
    missing: List[str] = []

    if not n["UNI"]:
        missing.append("UNI")
    if not n["TYPE"]:
        missing.append("TYPE")
    if not n["KEYWORD"]:
        missing.append("KEYWORD")

    return missing


def deterministic_followup_question(ner: Optional[Dict[str, Any]]) -> str:
    n = normalize_ner(ner)
    missing = missing_slots_for_followup(n)

    if not missing:
        return ""

    if missing[0] == "UNI":
        if n["TYPE"] and n["KEYWORD"]:
            return (
                f"어느 대학의 {n['TYPE'][0]} {n['KEYWORD'][0]} 정보를 확인해 드릴까요?"
            )
        return "어느 대학을 기준으로 확인해 드릴까요?"

    if missing[0] == "TYPE":
        if n["UNI"]:
            return (
                f"{n['UNI'][0]}의 수시와 정시 중 어떤 전형을 확인해 드릴까요?"
            )
        return "수시와 정시 중 어떤 전형을 확인해 드릴까요?"

    if missing[0] == "KEYWORD":
        if n["UNI"] and n["TYPE"]:
            return (
                f"{n['UNI'][0]} {n['TYPE'][0]}에서 궁금한 학과나 항목을 말씀해 주세요."
            )
        return "궁금한 학과나 모집 항목을 말씀해 주세요."

    return ""


def _flatten_rows(pair_to_rows: Dict[Tuple[str, str], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for values in (pair_to_rows or {}).values():
        rows.extend(values or [])
    return rows


def _append_followup_once(answer: str, followup: str) -> str:
    answer = (answer or "").strip()
    followup = (followup or "").strip()

    if not followup:
        return answer
    if followup in answer:
        return answer

    return f"{answer}\n\n{followup}".strip()


def decision_from_ner(ner: Optional[Dict[str, Any]]) -> str:
    """병합·규칙 보정이 끝난 최종 NER로 분류를 다시 계산한다."""
    n = normalize_ner(ner)
    if n["UNI"] and n["TYPE"] and n["KEYWORD"]:
        return "문서탐색"
    return "답변 생성"


def _retry_document_search_if_needed(
    result: Dict[str, Any],
    resolved_question: str,
    final_ner: Optional[Dict[str, Any]],
    uni_ex: UniExtractor,
    type_ex: TypeExtractor,
    kw_ex: KeywordExtractorBridge,
    api_key: str,
    gemini_model: str,
    llm_model: str,
) -> Dict[str, Any]:
    """
    명시 규칙으로 완성된 NER는 UNI/TYPE/KEYWORD를 모두 갖지만,
    answer_one 내부 NER 모델이 짧은 키워드를 놓쳐 일반 답변으로 분류한 경우
    한 번 더 모집요강 검색형 질문으로 재시도한다.
    """
    if decision_from_ner(final_ner) != "문서탐색":
        return result
    if (result.get("decision") or "").strip() == "문서탐색":
        return result

    n = normalize_ner(final_ner)
    canonical = " ".join(n["UNI"] + n["TYPE"] + n["KEYWORD"] + ["모집요강"]).strip()
    retry = answer_one(
        canonical or resolved_question,
        uni_ex, type_ex, kw_ex,
        api_key, gemini_model,
        llm_model=llm_model,
    )
    if (retry.get("decision") or "").strip() == "문서탐색":
        return retry
    return result


def run_single_turn(
    question: str,
    uni_ex: UniExtractor,
    type_ex: TypeExtractor,
    kw_ex: KeywordExtractorBridge,
    api_key: str,
    gemini_model: str,
    llm_model: str = "gpt-4o-mini",
):
    """
    main.py에서 호출하는 첫 질문 처리 함수.

    반환 형식:
    answer, decision, ner, rows, stats
    """
    question = (question or "").strip()
    current_ner = _extract_ner_direct(question, uni_ex, type_ex, kw_ex)

    if is_out_of_domain_or_injection(question, current_ner):
        return (
            build_domain_redirect_answer(question),
            "답변 생성",
            current_ner,
            [],
            {},
        )

    result = answer_one(
        question,
        uni_ex,
        type_ex,
        kw_ex,
        api_key,
        gemini_model,
        llm_model=llm_model,
    )

    result_ner = {
        "UNI": _slot_list(result.get("ner_uni")),
        "TYPE": _slot_list(result.get("ner_type")),
        "KEYWORD": _slot_list(result.get("ner_kw")),
    }

    # 직접 추출 결과가 더 잘 잡힌 슬롯은 보완한다.
    result_ner = merge_ner_context(result_ner, current_ner)

    result = _retry_document_search_if_needed(
        result, question, result_ner,
        uni_ex, type_ex, kw_ex, api_key, gemini_model, llm_model,
    )
    retry_ner = {
        "UNI": _slot_list(result.get("ner_uni")),
        "TYPE": _slot_list(result.get("ner_type")),
        "KEYWORD": _slot_list(result.get("ner_kw")),
    }
    result_ner = merge_ner_context(result_ner, retry_ner)

    missing = missing_slots_for_followup(result_ner)
    answer = (result.get("answer") or "").strip()
    decision = decision_from_ner(result_ner) if not missing else "재질문"

    if missing:
        followup = deterministic_followup_question(result_ner)
        answer = _append_followup_once(answer, followup)
        decision = "재질문"

    rows = _flatten_rows(result.get("pair_to_rows") or {})
    stats = result.get("stats") or {}

    return answer, decision, result_ner, rows, stats


AFFIRMATIVE_ONLY = {
    "응", "어", "네", "예", "넵", "좋아", "그래", "맞아",
    "알려줘", "해줘", "진행해줘", "확인해줘"
}


def is_affirmative_only(text: str) -> bool:
    normalized = re.sub(r"[\s.!?~]+", "", (text or "").strip().lower())
    return normalized in {re.sub(r"[\s.!?~]+", "", x.lower()) for x in AFFIRMATIVE_ONLY}


def build_followup_resolved_question(
    previous_user_question: str,
    user_input: str,
    merged_ner: Optional[Dict[str, Any]],
    current_ner: Optional[Dict[str, Any]],
) -> str:
    """
    question_1은 직전 사용자의 질문이고 question_2는 현재 사용자의 질문이다.

    이전 문맥은 전달받은 NER에서 복원하므로, 정상 상황에서는
    question_1을 다시 분석하거나 LLM에 전달하지 않는다.

    - 현재 입력에 새로운 정보가 있으면 현재 입력을 보존한다.
    - 현재 입력이 '응/네/알려줘'처럼 긍정만 있으면 병합된 NER로 질문을 만든다.
    - NER로 잡히지 않는 연도·조건은 현재 입력 원문을 함께 보존한다.
    """
    user_input = (user_input or "").strip()
    current = normalize_ner(current_ner)

    if is_affirmative_only(user_input):
        return build_resolved_question("알려줘", merged_ner)

    has_current_slot = any(current.values())
    if not has_current_slot and user_input:
        return build_resolved_question(user_input, merged_ner)

    return build_resolved_question(user_input, merged_ner)

def run_followup_turn(
    previous_text: str,
    user_input: str,
    prev_ner: Optional[Dict[str, Any]],
    uni_ex: UniExtractor,
    type_ex: TypeExtractor,
    kw_ex: KeywordExtractorBridge,
    api_key: str,
    gemini_model: str,
    llm_model: str = "gpt-4o-mini",
):
    """
    직전 사용자 질문(question_1), 현재 사용자 질문(question_2), 이전 NER를 연결한다.

    처리 우선순위:
    1) 프론트가 전달한 이전 NER를 즉시 사용한다.
    2) 현재 질문(question_2)만 NER 모델로 분석한다.
    3) 이전 NER와 현재 NER를 Python 코드로 병합한다.
    4) 이전 NER가 비어 있을 때만 question_1을 NER 모델로 분석한다.

    따라서 정상 요청에서는 question_1에 대한 추가 NER·GPT 호출이 발생하지 않는다.
    """
    previous_text = (previous_text or "").strip()
    user_input = (user_input or "").strip()

    if is_negative_only(user_input):
        return (
            "알겠습니다. 다른 대학 입시 정보가 궁금하실 때 말씀해 주세요.",
            "답변 생성",
            normalize_ner(prev_ner),
            [],
            {},
        )

    current_ner = _extract_ner_direct(user_input, uni_ex, type_ex, kw_ex)

    if is_out_of_domain_or_injection(user_input, current_ner):
        return (
            build_domain_redirect_answer(user_input),
            "답변 생성",
            current_ner,
            [],
            {},
        )

    # 이전 NER를 최우선 사용한다.
    # 정상적인 요청에서는 question_1(이전 사용자 질문)을 다시 분석하지 않는다.
    previous_ner = normalize_ner(prev_ner)
    used_question_1_fallback = False
    if not any(previous_ner.values()) and previous_text:
        previous_ner = _extract_ner_direct(previous_text, uni_ex, type_ex, kw_ex)
        used_question_1_fallback = True

    merged_ner = merge_ner_context(previous_ner, current_ner)
    resolved_question = build_followup_resolved_question(
        previous_text,
        user_input,
        merged_ner,
        current_ner,
    )

    # 현재 입력과 이전 NER 어느 쪽에도 입시 정보가 없다면 입시 챗봇 안내로 전환한다.
    if not is_admission_related(user_input, current_ner) and not any(merged_ner.values()):
        return (
            build_domain_redirect_answer(user_input),
            "답변 생성",
            merged_ner,
            [],
            {},
        )

    result = answer_one(
        resolved_question,
        uni_ex,
        type_ex,
        kw_ex,
        api_key,
        gemini_model,
        llm_model=llm_model,
    )

    result_ner = {
        "UNI": _slot_list(result.get("ner_uni")),
        "TYPE": _slot_list(result.get("ner_type")),
        "KEYWORD": _slot_list(result.get("ner_kw")),
    }

    # 재작성 문장에서 추출이 누락되더라도 이미 병합한 NER가 사라지지 않게 한다.
    final_ner = merge_ner_context(merged_ner, result_ner)

    result = _retry_document_search_if_needed(
        result, resolved_question, final_ner,
        uni_ex, type_ex, kw_ex, api_key, gemini_model, llm_model,
    )
    retry_ner = {
        "UNI": _slot_list(result.get("ner_uni")),
        "TYPE": _slot_list(result.get("ner_type")),
        "KEYWORD": _slot_list(result.get("ner_kw")),
    }
    final_ner = merge_ner_context(final_ner, retry_ner)

    missing = missing_slots_for_followup(final_ner)
    answer = (result.get("answer") or "").strip()
    decision = decision_from_ner(final_ner) if not missing else "재질문"

    if missing:
        followup = deterministic_followup_question(final_ner)
        answer = _append_followup_once(answer, followup)
        decision = "재질문"

    rows = _flatten_rows(result.get("pair_to_rows") or {})
    stats = result.get("stats") or {}
    stats["resolved_question"] = resolved_question
    stats["used_question_1_fallback"] = used_question_1_fallback

    return answer, decision, final_ner, rows, stats

def print_result_7lines(result: Dict[str, Any], dt: float) -> None:
    print(f"입력문장: {result['input']}")
    print(f"NER 추출 : UNI:{result['ner_uni']}  TYPE:{result['ner_type']}  KEYWORD:{result['ner_kw']}")
    print(f"최종 분류 : {result['decision']}")

    stats = result.get("stats") or {}
    pairs_n = stats.get("pairs", 0)
    docs_n = stats.get("docs_found", 0)
    pages_n = stats.get("pages_scored", 0)
    print(f"매칭쌍 : {pairs_n}개 , 문서 발견 : {docs_n}개 , 스코어링 대상 페이지 : {pages_n}장")

    pair_to_rows = result.get("pair_to_rows") or {}
    for (uni, typ), rows in pair_to_rows.items():
        rows_sorted = sorted(rows, key=lambda r: float(r.get("score", 0.0)), reverse=True)[:3]
        print(f"     ▷ 페어: [{uni} | {typ}]  (Top3)")
        for i, r in enumerate(rows_sorted, 1):
            doc = os.path.basename(r.get("doc_path", ""))
            p = r.get("page_index")
            sc = float(r.get("score", 0.0))
            kw = r.get("matched_keywords", "")
            print(f"       - Top{i}: {doc} | p.{p} | score={sc:.4f} | kw={kw}")

    print("챗봇 답변:")
    print(result.get("answer", ""))

    sources = result.get("sources") or []
    if sources:
        print("출처:")
        for s in sources:
            print(s)

    print(f"처리시간: {dt:.3f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", type=str, default="", help="단일 질문")
    ap.add_argument("--file", type=str, default="test.txt", help="질문 파일(기본: test.txt)")
    ap.add_argument("--llm_model", type=str, default="gpt-4o-mini")
    ap.add_argument("--top_pages", type=int, default=3)
    ap.add_argument("--quota_pages_per_pair", type=int, default=2)
    args = ap.parse_args()

    api_key, gemini_model = load_env()
    uni_ex = UniExtractor(max_len=128)
    type_ex = TypeExtractor()
    kw_ex = KeywordExtractorBridge(topn=10)

    if args.text.strip():
        q = args.text.strip()
        t0 = time.perf_counter()
        result = answer_one(
            q,
            uni_ex, type_ex, kw_ex,
            api_key, gemini_model,
            llm_model=args.llm_model,
            top_pages=args.top_pages,
            quota_pages_per_pair=args.quota_pages_per_pair,
        )
        dt = time.perf_counter() - t0
        print_result_7lines(result, dt)
        return

    file_path = args.file
    if not os.path.exists(file_path):
        print(f"질문 파일 없음: {file_path}")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]

    for q in questions:
        t0 = time.perf_counter()
        result = answer_one(
            q,
            uni_ex, type_ex, kw_ex,
            api_key, gemini_model,
            llm_model=args.llm_model,
            top_pages=args.top_pages,
            quota_pages_per_pair=args.quota_pages_per_pair,
        )
        dt = time.perf_counter() - t0
        print_result_7lines(result, dt)
        print()


if __name__ == "__main__":
    main()