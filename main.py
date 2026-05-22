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
import generate_answers as ga
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


def unique_extend(base: List[str], values: List[str]) -> List[str]:
    out = list(base)
    seen = set(out)

    for v in values:
        if v and v not in seen:
            out.append(v)
            seen.add(v)

    return out


def merge_ner_keyword(
    prev_ner: Optional[Dict[str, Any]],
    curr_ner: Optional[Dict[str, Any]],
) -> Dict[str, List[str]]:
    prev = make_ner_payload(prev_ner)
    curr = make_ner_payload(curr_ner)

    merged = {
        "UNI": [],
        "TYPE": [],
        "KEYWORD": [],
    }

    merged["UNI"] = unique_extend(merged["UNI"], curr["UNI"])
    merged["UNI"] = unique_extend(merged["UNI"], prev["UNI"])

    merged["TYPE"] = unique_extend(merged["TYPE"], curr["TYPE"])
    merged["TYPE"] = unique_extend(merged["TYPE"], prev["TYPE"])

    merged["KEYWORD"] = unique_extend(merged["KEYWORD"], prev["KEYWORD"])
    merged["KEYWORD"] = unique_extend(merged["KEYWORD"], curr["KEYWORD"])

    return merged


def build_query_from_ner(
    merged_ner: Dict[str, List[str]],
    fallback_question: str,
) -> str:
    parts: List[str] = []

    parts.extend(merged_ner.get("UNI", []))
    parts.extend(merged_ner.get("TYPE", []))
    parts.extend(merged_ner.get("KEYWORD", []))

    if parts:
        return " ".join(parts).strip() + " 알려줘"

    return fallback_question.strip()


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
            "NER_Keyword": ner_payload,
        }

    resp: Dict[str, Any] = {
        "answer": answer,
        "reply": False,
        "NER_Keyword": ner_payload,
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


@app.route("/answer", methods=["POST"])
def answer_endpoint():
    data = request.get_json(force=True) or {}

    q1: str = data.get("question_1", "") or ""
    q2: str = data.get("question_2", "") or ""
    first: bool = bool(data.get("first", True))
    prev_ner_keyword: Dict[str, Any] = data.get("NER_Keyword", {}) or {}

    first_q = q1.strip()
    follow_q = q2.strip()

    if first:
        if not first_q:
            return jsonify({"error": "question_1 이 비어 있습니다."}), 400

        answer, decision, ner, rows, stats = run_turn(first_q)
        resp = build_answer_response(answer, decision, rows, ner)
        return jsonify(resp)

    if not first_q:
        return jsonify({"error": "question_1 이 비어 있습니다."}), 400

    if not follow_q:
        prev_payload = make_ner_payload(prev_ner_keyword)

        combined_text = f"""[사용자 첫 질문]
{first_q}
"""

        user_prompt = ga.DIRECT_ANSWER_USER_TEMPLATE.format(
            question=combined_text
        )

        final_answer = ga.gpt_chat(
            ga.EXPERT_SYSTEM_PROMPT,
            user_prompt,
            model=MODEL_NAME,
        )

        return jsonify({
            "answer": final_answer,
            "reply": False,
            "NER_Keyword": prev_payload,
        })

    prev_payload = make_ner_payload(prev_ner_keyword)

    # 클라이언트가 NER_Keyword를 비워 보낸 경우 fallback으로 question_1 재분석
    if not any(prev_payload.values()):
        first_answer, first_decision, first_ner, first_rows, first_stats = run_turn(first_q)
        prev_payload = make_ner_payload(first_ner)

    # question_2 단독 NER 추출
    follow_answer, follow_decision, follow_ner, follow_rows, follow_stats = run_turn(follow_q)
    curr_payload = make_ner_payload(follow_ner)

    # 이전 NER_Keyword + 현재 question_2 NER 병합
    merged_ner = merge_ner_keyword(prev_payload, curr_payload)

    # 병합된 NER 기반 최종 질의 생성
    final_query = build_query_from_ner(merged_ner, follow_q)

    final_answer, final_decision, final_ner, final_rows, final_stats = run_turn(final_query)

    # 후속 질문의 상태값은 최종 NER보다 merged_ner를 우선 사용
    resp = build_answer_response(final_answer, final_decision, final_rows, merged_ner)

    return jsonify(resp)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)