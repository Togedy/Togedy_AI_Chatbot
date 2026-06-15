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
    "아래 질문에 대해 먼저 일반적인 입시 흐름/경향을 바탕으로 설명해 주세요.\n"
    "단정적인 수치/일정/규정은 피하고, 사용자가 다음 행동을 할 수 있게 방향을 잡아주세요.\n\n"
    "출력 규칙:\n"
    "1) 설명: 4~8문장\n"
    "2) 마지막: 더 정확한 안내를 위해 필요한 '정확한 키워드' 1~2개를 공손하게 질문 1문장으로 요청\n"
    "3) 금지 문구는 절대 쓰지 말 것(문서/컨텍스트/정확 확인 관련 면책 문구)\n\n"
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

        fu_prompt = build_followup_prompt(text, ner_uni, ner_type, ner_kw)
        followup = gpt_chat(
            system_prompt=EXPERT_SYSTEM_PROMPT,
            user_prompt=fu_prompt,
            model=llm_model,
            temperature=0.2,
        ).strip()

        answer_text = (main_answer + "\n\n" + followup).strip()
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

        fu_prompt = build_followup_prompt(text, ner_uni, ner_type, ner_kw)
        followup = gpt_chat(
            system_prompt=EXPERT_SYSTEM_PROMPT,
            user_prompt=fu_prompt,
            model=llm_model,
            temperature=0.2,
        ).strip()

        answer_text = (main_answer + "\n\n" + followup).strip()
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