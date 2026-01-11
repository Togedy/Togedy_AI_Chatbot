# -*- coding: utf-8 -*-
"""
generate_answers.py (업데이트 버전)
- NER 추출기 정상 연동
- 최종 분류(decision)에 따라 재질문/답변생성/문서탐색 분리
- 문서탐색:
    * 실제로 매칭된 페이지가 없으면 → 일반 GPT 답변으로 Fallback
    * 매칭된 페이지가 있으면 → RAG + 답변에 출처 포함

추가 규칙(2026-01):
- 콘솔 Top1~Top3 출력은 score와 무관하게 그대로 출력한다.
- 챗봇 답변의 출처/컨텍스트에는 score <= 0.01 인 항목을 포함하지 않는다.

추가 반영(2026-01 최신):
- 출처 표기 형식: "대학 , 타입 모집요강 p.10, p.72" 처럼 (대학, 타입)별로 한 줄에 페이지를 묶는다.
- 페이지 표기 순서는 score가 아니라 페이지 오름차순으로 한다.
- 페어가 여러 개(예: 정시+수시)인 경우, 특정 페어만 출처가 나오는 문제를 막기 위해 "페어별"로 출처를 구성한다.
- LLM이 출처 섹션을 일부만 남기거나 누락하더라도, 최종 출력은 후처리로 출처를 고정한다.
"""

from dotenv import load_dotenv
load_dotenv()

import os, sys, time, argparse
from typing import List, Dict, Any, Tuple
from statistics import mean

THIS = os.path.dirname(os.path.abspath(__file__))
if THIS not in sys.path:
    sys.path.insert(0, THIS)

# -----------------------
# 출처/컨텍스트 필터 기준
# -----------------------
MIN_SOURCE_SCORE = 0.01  # score > 0.01 인 페이지만 출처/컨텍스트에 포함

# --- NER 추출기 불러오기 ---
from extract_all import UniExtractor, TypeExtractor, KeywordExtractorBridge, load_env
from search_and_export import search_top_pages_for_query, read_questions, fmt_sec

# --- GPT 유틸 ---
def gpt_enabled() -> bool:
    return bool(os.getenv("OPENAI_API_KEY", "").strip())

def gpt_chat(
    system_prompt: str,
    user_prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
    max_tokens: int = 900,
) -> str:
    if not gpt_enabled():
        return "[알림] OPENAI_API_KEY가 설정되지 않아 모델 호출을 생략했습니다."
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"(GPT 호출 오류: {e})"

# --- 프롬프트 ---
EXPERT_SYSTEM_PROMPT = (
    "당신은 대한민국 대입 제도와 각 대학 모집요강에 정통한 입시 전문가입니다. "
    "최신 정보를 기준으로 간결하고 명확히 답하세요."
)

DOC_ANSWER_USER_TEMPLATE = """[질문]
{question}

[문서 컨텍스트]
{context}

[출처 후보]
{sources}

[지시사항]
- 위 컨텍스트 안에서만 근거를 찾아 답변하세요.
- 컨텍스트에 없는 내용은 추정하지 말고 '제시된 컨텍스트에 없음'이라고 하세요.
- 답변 마지막에 '출처:' 섹션을 추가하여 실제로 참고한 문서명을 1~3개 정도 bullet로 남기세요.
"""

# ▶ 답변 생성(비문서)용
DIRECT_ANSWER_USER_TEMPLATE = """[질문]
{question}

[지시사항]
- 제공된 모집요강 PDF를 사용하지 않고, 일반적인 대입 제도와 대학 입시 상식을 바탕으로 답변하세요.
- 가능하면 구체적인 예시나 조언을 간단히 제시하세요.
- 확실하지 않은 내용은 단정적으로 말하지 말고 '일반적인 기준으로는 ~'처럼 완곡하게 표현하세요.
"""

FOLLOWUP_USER_TEMPLATE = """[현재 파악된 정보]
- 대학: {uni}
- 전형: {type_}
- 키워드: {kw}

[지시사항]
- 빠진 정보를 물어보는 간결한 질문을 작성하세요.
"""

def build_followup_prompt(ner_uni, ner_type, ner_kw):
    u = ", ".join(ner_uni) if ner_uni else "(파악 안 됨)"
    t = (
        ", ".join(ner_type)
        if isinstance(ner_type, list) and ner_type
        else ner_type
        or "(파악 안 됨)"
    )
    k = ", ".join(ner_kw) if ner_kw else "(파악 안 됨)"
    return FOLLOWUP_USER_TEMPLATE.format(uni=u, type_=t, kw=k)

def _valid_source_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        r for r in rows
        if r.get("page_index", -1) != -1
        and float(r.get("score", 0.0)) > MIN_SOURCE_SCORE
    ]

def pick_context_from_rows(rows: List[Dict[str, Any]], topk: int = 3) -> str:
    """
    RAG 컨텍스트 블록 생성
    - score <= MIN_SOURCE_SCORE 인 페이지는 컨텍스트에서 제외
    - 페어가 여러 개인 경우 한쪽만 쏠리지 않게 "페어별 1개"를 우선 담고, 남는 자리는 score 순으로 채운다.
    """
    cands = _valid_source_rows(rows)
    if not cands:
        return "(컨텍스트 없음)"

    # (uni,type)별 그룹핑
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in cands:
        key = (r.get("matched_uni", ""), r.get("matched_type", ""))
        grouped.setdefault(key, []).append(r)

    # 페어별 1개씩 우선 선택(최고 점수)
    chosen: List[Dict[str, Any]] = []
    for key, lst in grouped.items():
        lst.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        chosen.append(lst[0])

    # 남은 후보들(중복 제거)
    chosen_ids = set((r.get("doc_path", ""), r.get("page_index", "")) for r in chosen)
    rest = [r for r in cands if (r.get("doc_path", ""), r.get("page_index", "")) not in chosen_ids]
    rest.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)

    final = chosen + rest
    final = final[:topk]

    blocks = []
    for i, r in enumerate(final, 1):
        src = os.path.basename(r.get("doc_path", "")) or "unknown.txt"
        page = r.get("page_index", "?")
        score = float(r.get("score", 0.0))
        snip = (r.get("snippet", "") or "").strip()
        blocks.append(f"[{i}] {src} p.{page} (score={score:.4f})\n{snip}")

    return "\n\n".join(blocks) if blocks else "(컨텍스트 없음)"

def build_sources_from_rows(rows: List[Dict[str, Any]], topk: int = 3) -> str:
    """
    출처 후보 문자열 생성 (요구사항 반영)
    - 함수 시그니처는 기존과 동일하게 topk 유지 (호환성)
    - (대학, 타입)별로 한 줄에 페이지를 묶어 표기
    - 각 (대학, 타입) 그룹 내에서 score 상위 topk개만 사용(너무 길어지는 것 방지)
    - 페이지 표기 순서는 score가 아니라 페이지 오름차순
    """
    cands = _valid_source_rows(rows)
    if not cands:
        return "(출처 후보 없음)"

    # (uni,type)별 그룹핑
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in cands:
        key = (r.get("matched_uni", ""), r.get("matched_type", ""))
        grouped.setdefault(key, []).append(r)

    lines: List[str] = []
    for (uni, type_), lst in grouped.items():
        # 대표성 확보: score 상위 topk개만 사용
        lst.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        lst = lst[:topk]

        # 페이지는 오름차순으로 출력 (중복 제거)
        pages = []
        for x in lst:
            p = x.get("page_index", None)
            try:
                p_int = int(p)
                pages.append(p_int)
            except Exception:
                continue
        pages = sorted(set(pages))

        if pages:
            page_str = ", ".join([f"p.{p}" for p in pages])
            lines.append(f"- {uni} , {type_} 모집요강 {page_str}")

    # 출력 순서 안정화: uni, type 정렬
    lines.sort()
    return "\n".join(lines) if lines else "(출처 후보 없음)"

def _strip_existing_sources(answer: str) -> str:
    """
    LLM이 만들어낸 '출처:' 섹션을 제거하고, 우리가 만든 출처로 고정하기 위한 전처리
    """
    if not answer:
        return ""
    idx = answer.rfind("출처:")
    if idx == -1:
        return answer.rstrip()
    return answer[:idx].rstrip()

def attach_fixed_sources(answer: str, sources: str) -> str:
    base = _strip_existing_sources(answer)
    if not sources or sources.strip() == "(출처 후보 없음)":
        return base
    return base + "\n\n출처:\n" + sources.strip()

def print_pairwise_top(rows: List[Dict[str, Any]], topk: int = 3):
    """
    콘솔 출력용 TopN
    - 요구사항: score에 상관없이 그대로 출력한다.
    """
    by_pair: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in rows:
        if r.get("page_index", -1) == -1:
            continue
        key = (r.get("matched_uni", ""), r.get("matched_type", ""))
        by_pair.setdefault(key, []).append(r)

    if not by_pair:
        print("     (검색 결과 없음)")
        return

    for (u, t), lst in by_pair.items():
        lst.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        print(f"     ▷ 페어: [{u or '-'} | {t or '-'}]  (Top{topk})")
        for i, r in enumerate(lst[:topk], 1):
            kw = r.get("matched_keywords") or "-"
            print(
                f"       - Top{i}: {os.path.basename(r['doc_path'])} | "
                f"p.{r['page_index']} | score={r['score']:.4f} | kw={kw}"
            )

# --- 메인 ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", default="test.txt")
    ap.add_argument("--model", default="gpt-4o-mini")
    args = ap.parse_args()

    api_key, gemini_model = load_env()
    uni_ex = UniExtractor(max_len=128)
    type_ex = TypeExtractor()
    kw_ex = KeywordExtractorBridge(topn=10)

    queries = read_questions(args.input)
    times: List[float] = []
    total_start = time.perf_counter()

    for idx, q in enumerate(queries, 1):
        t0 = time.perf_counter()

        rows, stats, ner = search_top_pages_for_query(
            q, uni_ex, type_ex, kw_ex, api_key, gemini_model, top_pages=3
        )

        decision = ner.get("decision")

        print(f"\n{idx}번 질문")
        print(f"입력문장: {q}")
        print(
            f"NER 추출 : UNI:{ner.get('uni')}  "
            f"TYPE:{ner.get('type')}  KEYWORD:{ner.get('keywords')}"
        )
        print(f"최종 분류 : {decision}")

        if decision == "문서탐색":
            print(
                f"매칭쌍 : {stats.get('pairs', 0)}개 , "
                f"문서 발견 : {stats.get('docs_found', 0)}개 , "
                f"스코어링 대상 페이지 : {stats.get('pages_scored', 0)}장"
            )
            print_pairwise_top(rows, topk=3)
        else:
            print("매칭쌍 : 0개 , 문서 발견 : 0개 , 스코어링 대상 페이지 : 0장")

        # -----------------------
        # 챗봇 답변 생성 분기
        # -----------------------
        if decision == "재질문":
            user_prompt = build_followup_prompt(
                ner.get("uni"), ner.get("type"), ner.get("keywords")
            )
            answer = gpt_chat(
                "너는 후속질문만 하는 한국어 비서다.", user_prompt, model=args.model
            )

        elif decision == "답변 생성":
            user_prompt = DIRECT_ANSWER_USER_TEMPLATE.format(question=q)
            answer = gpt_chat(EXPERT_SYSTEM_PROMPT, user_prompt, model=args.model)

        else:
            has_valid_page = any(
                r.get("page_index", -1) != -1 and float(r.get("score", 0.0)) > MIN_SOURCE_SCORE
                for r in rows
            )

            if has_valid_page:
                context = pick_context_from_rows(rows, topk=3)

                # 출처는 (uni,type)별로 묶어서 생성 (topk는 "페어별 최대 페이지 수"로 동작)
                sources = build_sources_from_rows(rows, topk=3)

                user_prompt = DOC_ANSWER_USER_TEMPLATE.format(
                    question=q, context=context, sources=sources
                )
                answer = gpt_chat(EXPERT_SYSTEM_PROMPT, user_prompt, model=args.model)

                # LLM이 출처를 누락/편집해도 최종 출력은 고정
                answer = attach_fixed_sources(answer, sources)
            else:
                fallback_prompt = DIRECT_ANSWER_USER_TEMPLATE.format(question=q)
                answer = gpt_chat(EXPERT_SYSTEM_PROMPT, fallback_prompt, model=args.model)

        dt = time.perf_counter() - t0
        times.append(dt)
        print("챗봇 답변:")
        print(answer)
        print(f"처리시간: {fmt_sec(dt)}")

    total_dt = time.perf_counter() - total_start
    print("\n=== 전체 처리 요약 ===")
    print(f"총 질의 수: {len(queries)}")
    print(f"총 처리시간: {fmt_sec(total_dt)}")
    print(f"평균 처리시간: {fmt_sec(mean(times)) if times else 0}")
    print(f"최대 처리시간: {fmt_sec(max(times)) if times else 0}")

if __name__ == "__main__":
    main()
