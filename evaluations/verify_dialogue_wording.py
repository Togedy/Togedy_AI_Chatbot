"""Capture complete API replies for clarification conversations.

Use --real-ner for local KoBERT inference. External LLM calls are forbidden:
these scenarios should use clarification templates and source-locked records.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import generate_answers as answers
import main


SCENARIOS = [
    ("대학 보충 후 문서 답변", ["정시 제출서류 알려줘", "연세대"], [True, False]),
    ("일부 보충 후 전체 양식 재입력", ["제출서류 알려줘", "연세대",
        "대학: 연세대\n전형: 정시\n학과·모집단위: 전체\n궁금한 내용: 제출서류"], [True, True, False]),
    ("대학 질문에 네라고 답변", ["정시 제출서류 알려줘", "네"], [True, True]),
    ("전형 보충", ["연세대 제출서류 알려줘", "정시요"], [True, False]),
    ("질문 항목 보충", ["연세대 정시", "제출서류요"], [True, False]),
    ("양식의 전형을 비워 둠", ["제출서류 알려줘",
        "대학: 연세대\n전형(수시·정시, 알면 세부 전형명도): \n학과·모집단위: 전체\n궁금한 내용: 제출서류"], [True, True]),
]


def run_scenarios(client):
    results = []
    for name, inputs, expected in SCENARIOS:
        turns, ner = [], {}
        for index, question in enumerate(inputs):
            body = {"first": index == 0, "question_1": inputs[index - 1] if index else question}
            if index:
                body.update(question_2=question, NER=ner)
            response = client.post("/answer", json=body)
            payload = response.get_json(silent=True) or {"error": f"Non-JSON HTTP {response.status_code}"}
            ok = response.status_code == 200 and payload.get("reply") is expected[index]
            if payload.get("reply") is False:
                ok = ok and payload.get("location", "").replace("\\", "/") == "university/yonsei/jungsi.pdf"
                ok = ok and "기본 학생부는 별도 제출 없이" in payload.get("answer", "")
            else:
                ok = ok and "location" not in payload and not any(k.startswith("NER_Page_") for k in payload)
            if index and expected[index] and "대학 보충" not in name:
                ok = ok and "한 메시지" in payload.get("answer", "")
            turns.append({"request": body, "status": response.status_code, "response": payload, "passed": bool(ok)})
            ner = payload.get("NER", {})
        results.append({"scenario": name, "passed": all(t["passed"] for t in turns), "turns": turns})
    return results


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-ner", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("Use a new output path; existing captures are not overwritten.")

    def reject_llm(**kwargs):
        raise AssertionError("These scenarios must not call an external LLM")

    with patch.object(answers, "call_llm", reject_llm):
        if args.real_ner:
            pipeline = main.get_pipeline()
            if not (pipeline.uni_ex.model is not None and callable(pipeline.type_ex._predict)
                    and pipeline.kw_ex.ke is not None and callable(pipeline.kw_ex.ke._model_predict)):
                raise RuntimeError("All three local NER models must be loaded for real-NER evaluation")
            # TYPE and KEYWORD adapters swallow errors; exercise their raw
            # prediction functions so missing models cannot look like success.
            pipeline.type_ex._predict("정시")
            pipeline.kw_ex.ke._model_predict("제출 서류")
            results = run_scenarios(main.create_app().test_client())
        else:
            pipeline = SimpleNamespace(uni_ex=None, type_ex=None, kw_ex=None,
                                       api_key="", gemini_model="", openai_model="")
            with patch.object(main, "get_pipeline", return_value=pipeline), patch.object(
                    answers, "_extract_ner_direct", return_value={"UNI": [], "TYPE": [], "KEYWORD": []}):
                results = run_scenarios(main.create_app().test_client())
    report = {"mode": "real_local_ner" if args.real_ner else "stubbed_ner", "external_llm": False,
              "passed": sum(r["passed"] for r in results), "total": len(results), "results": results}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, ensure_ascii=False))
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main_cli())
