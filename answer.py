# answer.py
# -*- coding: utf-8 -*-
"""
answer.py

역할
- 사용자로부터 질문을 입력받아 generate_answers.py에서 쓰는 파이프라인과 동일하게 처리.
- 첫 질문의 decision이 "재질문"인 경우:
    1) GPT가 만든 재질문 문장을 사용자에게 보여주고
    2) 사용자의 추가 입력을 받아
    3) 추가 입력에서 학교(UNI)가 감지되면:
         → 첫 질문 + 추가 입력을 합친 문장으로 다시 문서탐색/답변 생성
       학교가 끝까지 감지되지 않으면:
         → 문서탐색 없이 "답변 생성"만 수행
         (첫 질문, 재질문 문장, 사용자 추가 답변을 모두 포함)
"""

import os
import sys
import argparse
from typing import List, Dict, Any

THIS = os.path.dirname(os.path.abspath(__file__))
if THIS not in sys.path:
    sys.path.insert(0, THIS)

# --- generate_answers.py 안의 유틸/프롬프트 재사용 ---
import generate_answers as ga

# --- NER / 검색 유틸 ---
from extract_all import (
    UniExtractor,
    TypeExtractor,
    KeywordExtractorBridge,
    load_env,
)
from search_and_export import search_top_pages_for_query


def run_single_turn(
    question: str,
    uni_ex: UniExtractor,
    type_ex: TypeExtractor,
    kw_ex: KeywordExtractorBridge,
    api_key: str,
    gemini_model: str,
    model_name: str = "gpt-4o-mini",
):
    """
    generate_answers.py 의 한 질문 처리 로직을 함수화한 것.
    - 입력: question (문자열)
    - 출력: (answer:str, decision:str, ner:dict, rows:list, stats:dict)

    내부 로직은 generate_answers.py 의 main() 루프와 동일:
      - search_top_pages_for_query() 호출
      - decision(문서탐색/재질문/답변 생성)에 따라 분기
      - GPT 호출 시 ga.gpt_chat 및 ga.*_TEMPLATE 그대로 사용
    """
    # 1) 검색 + NER + 최종 분류
    rows, stats, ner = search_top_pages_for_query(
        question, uni_ex, type_ex, kw_ex, api_key, gemini_model, top_pages=3
    )
    decision = ner.get("decision")

    # 2) 분기 로직
    if decision == "재질문":
        user_prompt = ga.build_followup_prompt(
            question,
            ner.get("uni"),
            ner.get("type"),
            ner.get("keywords"),
        )
        answer = ga.gpt_chat(
            "너는 후속질문만 하는 한국어 비서다.",
            user_prompt,
            model=model_name,
        )

    elif decision == "답변 생성":
        user_prompt = ga.DIRECT_ANSWER_USER_TEMPLATE.format(question=question)
        answer = ga.gpt_chat(
            ga.EXPERT_SYSTEM_PROMPT,
            user_prompt,
            model=model_name,
        )

    else:
        # 문서탐색
        has_valid_page = any(r.get("page_index", -1) != -1 for r in rows)
        if has_valid_page:
            context = ga.pick_context_from_rows(rows, topk=3)
            sources = ga.build_sources_from_rows(rows, topk=3)
            user_prompt = ga.DOC_ANSWER_USER_TEMPLATE.format(
                question=question,
                context=context,
                sources=sources,
            )
            answer = ga.gpt_chat(
                ga.EXPERT_SYSTEM_PROMPT,
                user_prompt,
                model=model_name,
            )
        else:
            # 문서탐색 시도했지만 실제 매칭 페이지 없음 → 일반 답변으로 fallback
            fallback_prompt = ga.DIRECT_ANSWER_USER_TEMPLATE.format(question=question)
            answer = ga.gpt_chat(
                ga.EXPERT_SYSTEM_PROMPT,
                fallback_prompt,
                model=model_name,
            )

    return answer, decision, ner, rows, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()
    model_name = args.model

    # --- 환경 및 NER 모델 로드 (1회) ---
    api_key, gemini_model = load_env()
    uni_ex = UniExtractor(max_len=128)
    type_ex = TypeExtractor()
    kw_ex = KeywordExtractorBridge(topn=10)

    # 1) 첫 질문 입력
    first_q = input("질문을 입력하세요: ").strip()
    if not first_q:
        print("질문이 비어 있습니다. 종료합니다.")
        return

    # 2) 첫 질문 처리
    first_answer, first_decision, first_ner, first_rows, first_stats = run_single_turn(
        first_q, uni_ex, type_ex, kw_ex, api_key, gemini_model, model_name
    )

    # 2-1) 재질문이 아니라면 → 답변만 출력하고 종료
    if first_decision != "재질문":
        print("\n[답변]")
        print(first_answer)
        return

    # 3) 재질문인 경우
    print("\n[추가 질문]")
    print(first_answer)
    follow_q = input("추가 정보를 입력해 주세요: ").strip()

    # 추가 정보가 전혀 없으면 문서 탐색 없이 답변 생성
    if not follow_q:
        combined_text = f"[사용자 첫 질문]\n{first_q}\n"
        user_prompt = ga.DIRECT_ANSWER_USER_TEMPLATE.format(
            question=combined_text
        )
        final_answer = ga.gpt_chat(
            ga.EXPERT_SYSTEM_PROMPT,
            user_prompt,
            model=model_name,
        )
        print("\n[최종 답변]")
        print(final_answer)
        return

    # 4) 재질문(사용자 추가 입력)에 대해 NER만 따로 수행
    follow_uni = uni_ex.extract_uni(follow_q)
    follow_type = type_ex.extract_type(follow_q)
    follow_kw = kw_ex.extract_keywords(follow_q)

    # 4-1) 재질문에서 학교(UNI)가 감지된 경우
    if follow_uni:
        merged_q = first_q + "\n" + follow_q

        final_answer, final_decision, final_ner, final_rows, final_stats = run_single_turn(
            merged_q, uni_ex, type_ex, kw_ex, api_key, gemini_model, model_name
        )

        print("\n[최종 답변]")
        print(final_answer)
        return

    # 4-2) 재질문에서도 학교(UNI)가 감지되지 않은 경우
    combined_question_text = f"""[사용자 첫 질문]
{first_q}

[시스템이 추가로 물어본 질문]
{first_answer}

[사용자의 추가 답변]
{follow_q}
"""

    user_prompt = ga.DIRECT_ANSWER_USER_TEMPLATE.format(
        question=combined_question_text
    )
    final_answer = ga.gpt_chat(
        ga.EXPERT_SYSTEM_PROMPT,
        user_prompt,
        model=model_name,
    )

    print("\n[최종 답변]")
    print(final_answer)


if __name__ == "__main__":
    main()