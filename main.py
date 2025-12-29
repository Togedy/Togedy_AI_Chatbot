# main.py
# -*- coding: utf-8 -*-
"""
Flask 서버 버전

- 로직 기준은 answer.py와 동일하다.
- 재질문/문서탐색/답변생성 구분은 모두 기존 final_bucket 규칙
  (KEYWORD 유무 + UNI 유무)에 따라 동작한다.

요청(JSON) 예시:

1) 첫 질문 (first = true)
{
  "question_1": "서울대 수시에서의 모집인원과 연세대 정시에서 모집 일정에 대해 궁금하다",
  "question_2": "",
  "first": true,
  "NER_Keyword": {}
}

2) 재질문 이후 (first = false)
{
  "question_1": "수시에서의 모집인원에 대해 궁금하다",
  "question_2": "건국대에 대해 궁금합니다",
  "first": false,
  "NER_Keyword": {}
}

응답(JSON) 형식:

1) 재질문
{
  "answer": "어느 대학의 전형을 알고 싶으신가요?",
  "reply": true
}

2) 문서탐색 또는 답변 생성
{
  "answer": "...",
  "reply": false,
  "location": "university/konkuk/susi.pdf",
  "NER_Page_1": "p12",
  "NER_Page_2": "p13",
  "NER_Page_3": "p14"
}
※ 문서탐색이 아니거나 페이지가 없으면 location / NER_Page_*는 생략
"""

import os
import sys
from typing import List, Dict, Any, Tuple, Optional

from flask import Flask, request, jsonify

THIS = os.path.dirname(os.path.abspath(__file__))
if THIS not in sys.path:
    sys.path.insert(0, THIS)

# answer.py 의 run_single_turn 재사용
import answer as ans
import generate_answers as ga
from extract_all import (
    UniExtractor,
    TypeExtractor,
    KeywordExtractorBridge,
    load_env,
)

# --------------------------
# 전역 초기화 (모델/환경 1회 로드)
# --------------------------
api_key, gemini_model = load_env()
uni_ex = UniExtractor(max_len=128)
type_ex = TypeExtractor()
kw_ex = KeywordExtractorBridge(topn=10)

app = Flask(__name__)


# --------------------------
# 공통 유틸
# --------------------------
def extract_doc_meta(rows: List[Dict[str, Any]]) -> Tuple[Optional[str], List[int]]:
    """
    rows에서 최상위 문서 경로(location)와 상위 3개 페이지 번호 추출.
    location: best doc_path
    pages: [page1, page2, page3]
    """
    valid = [r for r in rows if r.get("page_index", -1) != -1]
    if not valid:
        return None, []

    # 점수 기준 내림차순 정렬
    valid.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
    best_doc = valid[0]["doc_path"]

    # 동일 문서에서 top3 페이지 가져오기
    same_doc = [r for r in valid if r["doc_path"] == best_doc]
    same_doc.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)

    pages: List[int] = []
    for r in same_doc:
        p = r.get("page_index")
        if isinstance(p, int) and p not in pages:
            pages.append(p)
        if len(pages) >= 3:
            break

    # 상대 경로 정제
    try:
        location = os.path.relpath(best_doc, THIS)
    except Exception:
        location = best_doc

    # 텍스트 파일명을 PDF로 변환
    # 예: university/konkuk/susi_text.txt → university/konkuk/susi.pdf
    if location.endswith("_text.txt"):
        location = location.replace("_text.txt", ".pdf")

    return location, pages


def build_answer_response(
    answer: str,
    decision: str,
    rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    최종 JSON 응답 형식 구성.
    - 재질문이면 reply=true, answer만.
    - 문서탐색/답변생성이면 reply=false,
      (문서탐색인 경우에만) location, NER_Page_i 추가.
    """
    # 재질문
    if decision == "재질문":
        return {
            "answer": answer,
            "reply": True,
        }

    # 기본: 최종 답변 (문서탐색 or 답변 생성)
    resp: Dict[str, Any] = {
        "answer": answer,
        "reply": False,
    }

    # 문서탐색인 경우에만 location / 페이지 정보 추가 (있을 때만)
    if decision == "문서탐색":
        location, pages = extract_doc_meta(rows)
        if location:
            resp["location"] = location
            for i, p in enumerate(pages, start=1):
                resp[f"NER_Page_{i}"] = f"p{p}"

    return resp


# --------------------------
# Flask 엔드포인트
# --------------------------
@app.route("/answer", methods=["POST"])
def answer_endpoint():
    """
    요청 JSON:
    {
      "question_1": "...",
      "question_2": "...",
      "first": true/false,
      "NER_Keyword": {...}
    }
    """
    data = request.get_json(force=True) or {}

    q1: str = data.get("question_1", "") or ""
    q2: str = data.get("question_2", "") or ""
    first: bool = bool(data.get("first", True))
    # NER_Keyword는 현재 사용하지 않지만 형식만 유지
    # ner_kw = data.get("NER_Keyword", {}) or {}

    model_name = "gpt-4o-mini"

    # ----------------------
    # 1) 첫 질문(first = true)
    # ----------------------
    if first:
        if not q1.strip():
            return jsonify({"error": "question_1 이 비어 있습니다."}), 400

        # answer.py 의 1턴 로직 그대로 사용
        answer, decision, ner, rows, stats = ans.run_single_turn(
            q1.strip(), uni_ex, type_ex, kw_ex, api_key, gemini_model, model_name
        )
        resp = build_answer_response(answer, decision, rows)
        return jsonify(resp)

    # ----------------------
    # 2) 재질문 이후(first = false)
    #    answer.py 의 main() 에 있는 2턴 로직을 HTTP로 옮긴 것
    # ----------------------
    first_q = q1.strip()
    follow_q = q2.strip()

    if not first_q:
        return jsonify({"error": "question_1 이 비어 있습니다."}), 400

    # (1) 첫 질문에 대해 다시 1턴 처리해서 first_answer/first_decision 확보
    first_answer, first_decision, first_ner, first_rows, first_stats = ans.run_single_turn(
        first_q, uni_ex, type_ex, kw_ex, api_key, gemini_model, model_name
    )

    # follow_q 가 비어 있으면 → 첫 질문만으로 답변 생성 (비문서)
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
        # 이 경우도 최종 답변이므로 reply=false
        return jsonify({
            "answer": final_answer,
            "reply": False,
        })

    # (2) 재질문(사용자 추가 입력)에 대해 NER 수행
    follow_uni = uni_ex.extract_uni(follow_q)
    # follow_type = type_ex.extract_type(follow_q)
    # follow_kw = kw_ex.extract_keywords(follow_q)

    # (2-1) 재질문에서 학교(UNI)가 감지된 경우
    if follow_uni:
        # "첫 질문 + 재질문" 을 합친 문장으로 다시 run_single_turn
        merged_q = first_q + "\n" + follow_q

        final_answer, final_decision, final_ner, final_rows, final_stats = ans.run_single_turn(
            merged_q, uni_ex, type_ex, kw_ex, api_key, gemini_model, model_name
        )

        resp = build_answer_response(final_answer, final_decision, final_rows)
        return jsonify(resp)

    # (2-2) 재질문에서도 학교(UNI)가 감지되지 않은 경우
    #  → 문서 탐색 포기, "답변 생성"으로만 처리
    #  (첫 질문 + 첫 답변(재질문 문장) + 추가 입력을 모두 포함)
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

    return jsonify({
        "answer": final_answer,
        "reply": False,
    })


if __name__ == "__main__":
    # python main.py 로 실행
    app.run(host="0.0.0.0", port=5000, debug=True)
