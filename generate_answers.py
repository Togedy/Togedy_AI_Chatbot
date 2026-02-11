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


def call_llm(prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.1) -> str:
    if OpenAI is None:
        return "LLM 라이브러리(openai)가 설치되어 있지 않아 답변을 생성할 수 없습니다."
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return "OPENAI_API_KEY 환경변수가 설정되어 있지 않아 답변을 생성할 수 없습니다."

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "너는 한국 대학 입시 모집요강 텍스트를 기반으로 답하는 도우미다. 제공된 문서 내용 밖의 추측을 금지한다."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


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
        else:
            out.append(f"- {uni} , {typ} 모집요강 (페이지 정보 없음)")
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
        "아래는 대학 모집요강 텍스트(페이지 발췌)이다.\n"
        "반드시 제공된 텍스트 안의 정보만 근거로 답하라. 추측 금지.\n\n"
        f"질문: {question}\n"
        f"대학/전형: {uni} / {typ}\n"
        f"대상 학과/학부: {majors_str}\n"
        f"참고 페이지: {pages_info}\n\n"
        "요구사항:\n"
        "1) 대상 학과/학부의 모집인원 '합계'를 우선 제시하라.\n"
        "2) 표에 전형별 항목(예: 지역균형/일반전형/기회균형특별전형 등)이 함께 있으면 전형별 인원도 함께 제시하라.\n"
        "3) 표에서 확인할 수 없으면 '제공 문서에서 확인 불가'라고 말하라.\n"
        "4) 가능한 한 간단히, 숫자 근거가 되는 열/항목을 짧게 언급하라.\n\n"
        "제공 문서:\n"
        "-----\n"
        f"{doc_join}\n"
        "-----\n"
    )


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

    decision = ner.get("decision", "")

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

    answer_text = ""
    sources_lines: List[str] = []

    if decision != "문서탐색":
        answer_text = "추가 정보가 필요합니다."
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
        lines = ["모집 인원은 다음과 같습니다:\n"]
        selected_pages: Dict[Tuple[str, str], List[int]] = {}

        for (uni, typ), prows in pair_to_rows.items():
            picks = pick_best_quota_pages(prows, majors, max_pages=quota_pages_per_pair)
            if not picks:
                lines.append(f"- {uni} {typ}: 제공 문서에서 확인 불가")
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
                lines.append(f"- {uni} {typ}: 제공 문서에서 확인 불가")
                continue

            prompt = build_quota_prompt(text, uni, typ, majors, page_texts)
            quota_ans = call_llm(prompt, model=llm_model, temperature=0.0)
            lines.append(f"- {uni} {typ}: {quota_ans}")

        answer_text = "\n".join(lines).strip()
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

    context_blocks: List[str] = []
    for (uni, typ), prows in pair_to_rows.items():
        prows_sorted = sorted(prows, key=lambda r: int(r.get("page_index", 10**9)))
        snippet_join = "\n".join([f"[p.{r.get('page_index')}] {r.get('snippet','')}" for r in prows_sorted])
        context_blocks.append(f"({uni}, {typ}) 컨텍스트:\n{snippet_join}")

    prompt = (
        f"질문:\n{text}\n\n"
        "컨텍스트(모집요강 발췌):\n"
        + "\n\n".join(context_blocks)
        + "\n\n"
        "요구사항:\n"
        "- 컨텍스트에 있는 내용만 사용\n"
        "- 표/숫자는 추측하지 말고, 컨텍스트에 근거가 없으면 '컨텍스트에서 확인 불가'라고 말할 것\n"
    )

    answer_text = call_llm(prompt, model=llm_model, temperature=0.1)
    sources_lines = build_sources(pair_to_rows)

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

    print("출처:")
    for s in (result.get("sources") or []):
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
