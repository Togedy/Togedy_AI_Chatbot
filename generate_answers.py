# -*- coding: utf-8 -*-
"""
generate_answers.py (최종 버전)
- 각 질문별 상세 매칭쌍 출력
- 전체 처리 요약에 평균/최대 처리시간 추가
"""

import os, sys, time, argparse
from typing import List, Dict, Any
from statistics import mean

THIS = os.path.dirname(os.path.abspath(__file__))
if THIS not in sys.path:
    sys.path.insert(0, THIS)

from extract_all import UniExtractor, TypeExtractor, KeywordExtractorBridge, load_env
from search_and_export import search_top_pages_for_query, read_questions, fmt_sec

# GPT 호출 함수 (생략된 부분 동일)
from openai import OpenAI
def gpt_chat(client, system_prompt, user_prompt, model="gpt-4o-mini", temperature=0.3, max_tokens=800):
    try:
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

EXPERT_SYSTEM_PROMPT = (
    "당신은 대한민국 대입 제도와 각 대학 모집요강에 정통한 입시 전문가입니다. "
    "항상 최신 모집요강을 기준으로 정확하고 간결하게 설명하고, "
    "수험생이 바로 실행할 수 있는 체크리스트를 제공합니다."
)
DOC_ANSWER_USER_TEMPLATE = "[질문]\n{question}\n\n[문서 컨텍스트]\n{context}\n\n[지시사항]\n- 위 컨텍스트 안에서만 답변하세요.\n- 컨텍스트에 없는 정보는 추정하지 마세요."
DIRECT_ANSWER_USER_TEMPLATE = "[질문]\n{question}\n\n입시 전문가처럼 간결하게 답하세요."
FOLLOWUP_USER_TEMPLATE = "[현재 파악된 정보]\n대학: {uni}\n전형: {type_}\n키워드: {kw}\n\n부족한 정보를 묻는 한 문장의 질문만 작성하세요."

def build_followup_prompt(ner_uni, ner_type, ner_kw):
    u = ", ".join(ner_uni) if ner_uni else "(파악 안 됨)"
    t = ", ".join(ner_type) if isinstance(ner_type, list) and ner_type else ner_type or "(파악 안 됨)"
    k = ", ".join(ner_kw) if ner_kw else "(파악 안 됨)"
    return FOLLOWUP_USER_TEMPLATE.format(uni=u, type_=t, kw=k)

def pick_context_from_rows(rows, topk=3):
    cands = [r for r in rows if r.get("page_index", -1) != -1]
    cands.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
    blocks = []
    for i, r in enumerate(cands[:topk], 1):
        src = os.path.basename(r.get("doc_path", "")) or "unknown.txt"
        p = r.get("page_index", "?")
        s = float(r.get("score", 0.0))
        kw = "|".join(r.get("keywords", [])) if r.get("keywords") else "-"
        blocks.append(f"- Top{i}: {src} | p.{p} | score={s:.4f} | kw={kw}")
    return "\n".join(blocks) if blocks else "(검색 결과 없음)"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", default="test.txt")
    ap.add_argument("--model", default="gpt-4o-mini")
    args = ap.parse_args()

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
    uni_ex, type_ex, kw_ex = UniExtractor(), TypeExtractor(), KeywordExtractorBridge()

    queries = read_questions(args.input)
    times = []
    total_t0 = time.perf_counter()

    for idx, q in enumerate(queries, 1):
        t0 = time.perf_counter()
        rows, stats, ner = search_top_pages_for_query(q, uni_ex, type_ex, kw_ex, "", "", top_pages=5)
        decision = ner.get("decision")

        if decision == "재질문":
            user_prompt = build_followup_prompt(ner.get("uni"), ner.get("type"), ner.get("keywords"))
            answer = gpt_chat(client, "너는 간결히 후속질문만 하는 비서다.", user_prompt)
        elif decision == "답변 생성":
            user_prompt = DIRECT_ANSWER_USER_TEMPLATE.format(question=q)
            answer = gpt_chat(client, EXPERT_SYSTEM_PROMPT, user_prompt)
        else:
            context = pick_context_from_rows(rows)
            user_prompt = DOC_ANSWER_USER_TEMPLATE.format(question=q, context=context)
            answer = gpt_chat(client, EXPERT_SYSTEM_PROMPT, user_prompt)

        dt = time.perf_counter() - t0
        times.append(dt)

        print(f"\n{idx}번 질문")
        print(f"입력문장: {q}")
        print(f"NER 추출 : UNI:{ner.get('uni')}  TYPE:{ner.get('type')}  KEYWORD:{ner.get('keywords')}")
        print(f"최종 분류 : {decision}")
        print(f"매칭쌍 : {stats.get('pairs', 0)}개 , 문서 발견 : {stats.get('docs', 0)}개 , 스코어링 대상 페이지 : {stats.get('pages', 0)}장")
        print(pick_context_from_rows(rows))
        print("챗봇 답변:")
        print(answer)
        print(f"처리시간: {fmt_sec(dt)}")

    total_dt = time.perf_counter() - total_t0
    avg_time = mean(times)
    max_time = max(times)

    print("\n=== 전체 처리 요약 ===")
    print(f"총 질의 수: {len(queries)}")
    print(f"총 처리시간: {fmt_sec(total_dt)}")
    print(f"평균 처리시간: {fmt_sec(avg_time)}")
    print(f"최대 처리시간: {fmt_sec(max_time)}")

if __name__ == "__main__":
    main()
