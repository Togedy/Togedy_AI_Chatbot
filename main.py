# main.py
# -*- coding: utf-8 -*-

import os
import sys
from typing import List, Dict, Any, Tuple, Optional

from flask import Flask, request, jsonify

THIS = os.path.dirname(os.path.abspath(__file__))
if THIS not in sys.path:
    sys.path.insert(0, THIS)

import answer as ans
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
        "UNI": _normalize_ner_list(ner.get("UNI", ner.get("uni"))),
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


def build_answer_response(
    answer: str,
    decision: str,
    rows: List[Dict[str, Any]],
    ner: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
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

            for i, p in enumerate(pages, start=1):
                resp[f"NER_Page_{i}"] = f"p{p}"

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
    bot_question: str,
    user_input: str,
    prev_ner: Optional[Dict[str, Any]],
):
    return ans.run_followup_turn(
        bot_question,
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

    # first=false일 때의 의미:
    # question_1 = 직전 챗봇 질문
    # question_2 = 사용자의 현재 입력
    if not question_1:
        return jsonify({"error": "question_1 이 비어 있습니다."}), 400

    if not question_2:
        return jsonify({
            "answer": "추가로 궁금한 내용을 입력해 주시면 이어서 안내해 드릴게요.",
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
    app.run(host="0.0.0.0", port=5000, debug=True)
