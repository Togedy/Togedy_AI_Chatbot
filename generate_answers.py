# -*- coding: utf-8 -*-
"""
generate_answers.py (오류 완전 제거 + NER 추출기 정상 연동 + 결정타입별 동작 분리)
"""
from dotenv import load_dotenv
load_dotenv()

import os, sys, time, argparse
from typing import List, Dict, Any
from statistics import mean

THIS = os.path.dirname(os.path.abspath(__file__))
if THIS not in sys.path:
    sys.path.insert(0, THIS)

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

[지시사항]
- 위 컨텍스트 안에서만 근거를 찾아 답변하세요.
- 컨텍스트에 없는 내용은 추정하지 말고 '제시된 컨텍스트에 없음'이라고 하세요.
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

def pick_context_from_rows(rows: List[Dict[str, Any]], topk: int = 3) -> str:
    cands = [r for r in rows if r.get("page_index", -1) != -1]
    cands.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
    blocks = []
    for i, r in enumerate(cands[:topk], 1):
        src = os.path.basename(r.get("doc_path", "")) or "unknown.txt"
        page = r.get("page_index", "?")
        score = r.get("score", 0.0)
        snip = r.get("snippet", "").strip()
        blocks.append(f"[{i}] {src} p.{page} (score={score:.4f})\n{snip}")
    return "\n\n".join(blocks) if blocks else "(컨텍스트 없음)"

def print_pairwise_top(rows: List[Dict[str, Any]], topk: int = 3):
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

    # ✅ NER 추출기 및 환경 불러오기
    api_key, gemini_model = load_env()
    uni_ex = UniExtractor(max_len=128)
    type_ex = TypeExtractor()
    kw_ex = KeywordExtractorBridge(topn=10)

    queries = read_questions(args.input)
    times: List[float] = []
    total_start = time.perf_counter()

    for idx, q in enumerate(queries, 1):
        t0 = time.perf_counter()

        # ✅ 통합 파이프라인: 여기서 NER + 최종 분류 + (필요 시) 문서 검색까지 수행
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

        # ▶ 문서 탐색일 때만 매칭쌍/문서 통계를 사용
        if decision == "문서탐색":
            print(
                f"매칭쌍 : {stats.get('pairs', 0)}개 , "
                f"문서 발견 : {stats.get('docs_found', 0)}개 , "
                f"스코어링 대상 페이지 : {stats.get('pages_scored', 0)}장"
            )
            print_pairwise_top(rows, topk=3)
        else:
            # 재질문 / 답변생성인 경우에는 검색 결과를 사용하지 않음
            print("매칭쌍 : 0개 , 문서 발견 : 0개 , 스코어링 대상 페이지 : 0장")

        # -----------------------
        # 챗봇 답변 생성 분기
        # -----------------------
        if decision == "재질문":
            # 1) 재질문: 매칭쌍/문서 사용 X, 바로 후속 질문 생성
            user_prompt = build_followup_prompt(
                ner.get("uni"), ner.get("type"), ner.get("keywords")
            )
            answer = gpt_chat(
                "너는 후속질문만 하는 한국어 비서다.", user_prompt, model=args.model
            )

        elif decision == "답변 생성":
            # 2) 답변 생성: 매칭쌍/문서 사용 X, 질문만으로 답변 생성
            user_prompt = DIRECT_ANSWER_USER_TEMPLATE.format(question=q)
            answer = gpt_chat(EXPERT_SYSTEM_PROMPT, user_prompt, model=args.model)

        else:
            # 3) 기본값(또는 '문서탐색'): 문서 컨텍스트 기반 RAG 답변
            context = pick_context_from_rows(rows, topk=3)
            user_prompt = DOC_ANSWER_USER_TEMPLATE.format(
                question=q, context=context
            )
            answer = gpt_chat(EXPERT_SYSTEM_PROMPT, user_prompt, model=args.model)

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
