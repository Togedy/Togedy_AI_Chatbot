# main.py
# -*- coding: utf-8 -*-

import os
import sys
import re
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


def unique_extend(base: List[str], values: List[str]) -> List[str]:
    out = list(base)
    seen = set(out)

    for v in values:
        if v and v not in seen:
            out.append(v)
            seen.add(v)

    return out


def merge_ner(
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


def clean_rewritten_question(text: str) -> str:
    text = text.strip()

    text = re.sub(r"^```(?:text|json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    prefixes = [
        "재작성된 질문:",
        "최종 질문:",
        "질문:",
        "resolved_question:",
    ]

    for p in prefixes:
        if text.startswith(p):
            text = text[len(p):].strip()

    text = text.strip('"').strip("'").strip()

    return text


def rewrite_followup_question(bot_question: str, user_input: str) -> str:
    """
    first=false일 때 사용.
    question_1 = 직전 챗봇 질문
    question_2 = 사용자 입력

    두 문장을 바탕으로 실제 서버가 처리해야 할 질문을 1문장으로 재작성한다.
    """

    system_prompt = """너는 대학 입시 챗봇의 후속 입력 재작성기다.
직전 챗봇 질문과 사용자 입력을 보고, 서버가 실제로 답변해야 할 질문을 한국어 한 문장으로 재작성한다.
설명하지 말고 최종 질문 한 문장만 출력한다."""

    user_prompt = f"""
[직전 챗봇 질문]
{bot_question}

[사용자 입력]
{user_input}

규칙:
1. 사용자가 "어", "응", "네", "알려줘", "좋아"처럼 답하면 직전 챗봇 질문에서 물어본 정보를 요청한 것으로 재작성한다.
2. 사용자가 직전 챗봇 질문에 조건을 추가하면 두 내용을 합쳐서 재작성한다.
3. 사용자가 완전히 다른 주제를 말하면 사용자 입력을 새 질문으로 재작성한다.
4. 사용자가 부정하면 "추가 질문 없음"이라고만 출력한다.
5. 출력은 반드시 최종 질문 한 문장만 한다.

예시:
직전 챗봇 질문: 수능 날짜에 대한 정보가 필요하신가요?
사용자 입력: 어
출력: 수능 날짜에 대한 정보를 알려줘

직전 챗봇 질문: 수능 날짜에 대한 정보가 필요하신가요?
사용자 입력: 2026년 기준으로 알려줘
출력: 2026년 수능 날짜에 대한 정보를 알려줘

직전 챗봇 질문: 수능 날짜에 대한 정보가 필요하신가요?
사용자 입력: 나는 건국대 컴퓨터공학부 입시 정보가 궁금해
출력: 건국대 컴퓨터공학부 입시 정보를 알려줘
"""

    raw = ga.gpt_chat(
        system_prompt,
        user_prompt,
        model=MODEL_NAME,
    )

    rewritten = clean_rewritten_question(raw)

    if not rewritten:
        return user_input.strip()

    return rewritten


@app.route("/answer", methods=["POST"])
def answer_endpoint():
    data = request.get_json(force=True) or {}

    q1: str = data.get("question_1", "") or ""
    q2: str = data.get("question_2", "") or ""

    # 프론트에서 NER 또는 NER_Keyword 둘 다 올 수 있게 처리
    prev_ner: Dict[str, Any] = (
        data.get("NER")
        or data.get("NER_Keyword")
        or {}
    )

    first: bool = bool(data.get("first", True))

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
        prev_payload = make_ner_payload(prev_ner)

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
            "NER": prev_payload,
        })

    prev_payload = make_ner_payload(prev_ner)

    # first=false일 때:
    # question_1 = 직전 챗봇 질문
    # question_2 = 사용자 입력
    rewritten_question = rewrite_followup_question(first_q, follow_q)

    if rewritten_question == "추가 질문 없음":
        return jsonify({
            "answer": "알겠습니다. 다른 궁금한 입시 정보가 있으면 질문해 주세요.",
            "reply": False,
            "NER": prev_payload,
        })

    # 재작성된 질문으로 먼저 실행
    answer, decision, ner, rows, stats = run_turn(rewritten_question)

    curr_payload = make_ner_payload(ner)

    # 이전 NER가 있으면 보강
    if any(prev_payload.values()):
        merged_ner = merge_ner(prev_payload, curr_payload)

        if any(merged_ner.values()):
            final_query = build_query_from_ner(merged_ner, rewritten_question)
            final_answer, final_decision, final_ner, final_rows, final_stats = run_turn(final_query)
            resp = build_answer_response(final_answer, final_decision, final_rows, merged_ner)
            return jsonify(resp)

    resp = build_answer_response(answer, decision, rows, curr_payload)
    return jsonify(resp)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)