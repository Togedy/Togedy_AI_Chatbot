# main.py
# -*- coding: utf-8 -*-

import os
import sys
from typing import List, Dict, Any, Tuple, Optional

from flask import Flask, request, jsonify

THIS = os.path.dirname(os.path.abspath(__file__))
if THIS not in sys.path:
    sys.path.insert(0, THIS)

import generate_answers as ans
from extract_all import (
    UniExtractor,
    TypeExtractor,
    KeywordExtractorBridge,
    load_env,
)

api_key, gemini_model = load_env()
uni_ex = UniExtractor(max_len=128)
type_ex = TypeExtractor()
kw_ex = KeywordExtractorBridge(topn=10)

app = Flask(__name__)
app.json.ensure_ascii = False
MODEL_NAME = "gpt-4o-mini"


def _normalize_ner_list(x: Any) -> List[str]:
    if x is None:
        return []

    if isinstance(x, list):
        return [str(v).strip() for v in x if str(v).strip()]

    s = str(x).strip()
    return [s] if s else []


def make_ner_payload(ner: Optional[Dict[str, Any]]) -> Dict[str, List[str]]:
    ner = ner or {}

    return {
        "UNI": _normalize_ner_list(ner.get("UNI", ner.get("uni", ner.get("UNT")))),
        "TYPE": _normalize_ner_list(ner.get("TYPE", ner.get("type"))),
        "KEYWORD": _normalize_ner_list(ner.get("KEYWORD", ner.get("keywords"))),
    }


def extract_doc_meta(rows: List[Dict[str, Any]]) -> Tuple[Optional[str], List[int]]:
    valid = [r for r in rows if r.get("page_index", -1) != -1]

    if not valid:
        return None, []

    valid.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
    best_doc = valid[0]["doc_path"]

    same_doc = [r for r in valid if r["doc_path"] == best_doc]
    same_doc.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)

    pages: List[int] = []

    for r in same_doc:
        p = r.get("page_index")

        if isinstance(p, int) and p not in pages:
            pages.append(p)

        if len(pages) >= 3:
            break

    try:
        location = os.path.relpath(best_doc, THIS)
    except Exception:
        location = best_doc

    if location.endswith("_text.txt"):
        location = location.replace("_text.txt", ".pdf")

    return location, pages


def extract_doc_metas(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    여러 대학/전형이 검색된 경우 문서별 위치와 상위 페이지를 반환한다.

    기존 location/NER_Page_* 필드는 하위 호환을 위해 유지하고,
    다중 대학 응답에서는 documents 배열을 추가한다.
    """
    valid = [r for r in rows if r.get("page_index", -1) != -1 and r.get("doc_path")]
    if not valid:
        return []

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in valid:
        grouped.setdefault(str(row["doc_path"]), []).append(row)

    documents: List[Dict[str, Any]] = []
    for doc_path, doc_rows in grouped.items():
        doc_rows.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)

        pages: List[int] = []
        for row in doc_rows:
            page = row.get("page_index")
            if isinstance(page, int) and page not in pages:
                pages.append(page)
            if len(pages) >= 3:
                break

        try:
            location = os.path.relpath(doc_path, THIS)
        except Exception:
            location = doc_path
        if location.endswith("_text.txt"):
            location = location.replace("_text.txt", ".pdf")

        best = doc_rows[0]
        documents.append({
            "UNI": str(best.get("matched_uni", "") or ""),
            "TYPE": str(best.get("matched_type", "") or ""),
            "location": location,
            "pages": [f"p{page}" for page in pages],
            "score": float(best.get("score", 0.0)),
        })

    documents.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return documents


def extract_followup_question(answer: str, decision: str) -> str:
    """
    재질문 응답의 마지막 비어 있지 않은 줄을 꼬리 질문으로 반환한다.

    generate_answers.py에서는 재질문을 항상 답변의 마지막 줄에 붙이므로
    프론트는 이 값을 다음 요청의 question_1로 그대로 전달하면 된다.
    """
    if decision != "재질문":
        return ""

    lines = [
        line.strip()
        for line in (answer or "").splitlines()
        if line.strip()
    ]
    return lines[-1] if lines else ""


def build_answer_response(
    answer: str,
    decision: str,
    rows: List[Dict[str, Any]],
    ner: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    기존 Spring 서버의 AiAnswerResponse DTO와 호환되는 필드만 반환한다.

    반환 허용 필드:
    - answer
    - reply
    - NER
    - location
    - NER_Page_1~3

    followup_question, documents처럼 서버 DTO에 없는 필드는 반환하지 않는다.
    다중 대학 질문도 answer와 NER에는 모두 반영되지만, 문서 메타데이터는
    기존 규격에 맞춰 가장 점수가 높은 문서 1개만 location/NER_Page_*로 반환한다.
    """
    ner_payload = make_ner_payload(ner)

    if decision == "재질문":
        return {
            "answer": answer,
            "reply": True,
            "NER": ner_payload,
        }

    resp: Dict[str, Any] = {
        "answer": answer,
        "reply": False,
        "NER": ner_payload,
    }

    if decision == "문서탐색":
        location, pages = extract_doc_meta(rows)

        if location:
            resp["location"] = location
            for i, page in enumerate(pages, start=1):
                resp[f"NER_Page_{i}"] = f"p{page}"

    return resp

def run_turn(question: str):
    return ans.run_single_turn(
        question,
        uni_ex,
        type_ex,
        kw_ex,
        api_key,
        gemini_model,
        MODEL_NAME,
    )


def run_followup_turn(
    previous_user_question: str,
    user_input: str,
    prev_ner: Optional[Dict[str, Any]],
):
    return ans.run_followup_turn(
        previous_user_question,
        user_input,
        prev_ner,
        uni_ex,
        type_ex,
        kw_ex,
        api_key,
        gemini_model,
        MODEL_NAME,
    )


@app.route("/answer", methods=["POST"])
def answer_endpoint():
    data = request.get_json(force=True) or {}

    q1: str = data.get("question_1", "") or ""
    q2: str = data.get("question_2", "") or ""
    first: bool = bool(data.get("first", True))

    # 기존 서버 연결 방식 유지:
    # 프론트가 NER 또는 NER_Keyword 중 어떤 이름으로 보내도 수용한다.
    prev_ner: Dict[str, Any] = (
        data.get("NER")
        or data.get("NER_Keyword")
        or {}
    )

    question_1 = q1.strip()
    question_2 = q2.strip()

    if first:
        if not question_1:
            return jsonify({"error": "question_1 이 비어 있습니다."}), 400

        answer, decision, ner, rows, stats = run_turn(question_1)
        resp = build_answer_response(answer, decision, rows, ner)
        return jsonify(resp)

    # first=false일 때:
    # question_1 = 직전 사용자의 질문
    # question_2 = 현재 사용자의 질문
    # NER = 직전 서버 응답에서 받은 이전 질문의 NER
    #
    # 응답 속도를 위해 전달받은 NER를 가장 먼저 사용한다.
    # NER가 정상적으로 전달되면 question_1은 다시 분석하지 않는다.
    # NER가 비어 있을 때만 question_1을 fallback으로 분석한다.
    if not question_1:
        return jsonify({"error": "question_1 이 비어 있습니다."}), 400

    if not question_2:
        return jsonify({
            "answer": "추가로 궁금한 내용을 입력해 주시면 이어서 안내해 드릴게요.",
            "followup_question": question_1,
            "reply": True,
            "NER": make_ner_payload(prev_ner),
        })

    answer, decision, ner, rows, stats = run_followup_turn(
        question_1,
        question_2,
        prev_ner,
    )

    resp = build_answer_response(answer, decision, rows, ner)
    return jsonify(resp)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False
    )
