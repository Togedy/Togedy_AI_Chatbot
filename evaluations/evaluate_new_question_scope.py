"""Evaluate new question interpretation and reviewed-record filtering locally.

This is a deterministic component benchmark, not an end-to-end LLM accuracy
score. Questions and expected scope are frozen before the implementation fix.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from question_plan import build_question_plan
from llm.reviewed_facts import verified_records, matches, discovery_answer


def evaluate(case, records):
    plan = build_question_plan(case["question"])
    actual_records = sorted(r["id"] for r in records if plan.intent == "discovery" and matches(r, plan))
    actual = {"intent": plan.intent, "records": actual_records,
              "universities": plan.universities, "rounds": plan.rounds,
              "tracks": plan.tracks, "topics": plan.topics,
              "excluded_universities": plan.constraints.get("excluded_universities", []),
              "excluded_majors": plan.constraints.get("excluded_majors", [])}
    if "shortcut" in case:
        actual["shortcut"] = discovery_answer(plan) is not None
    failures = {}
    for key, expected in case.items():
        if key == "question":
            continue
        value = actual.get(key)
        if isinstance(expected, list):
            success = set(expected) == set(value or []) if key not in ("topics", "tracks") else set(expected) <= set(value or [])
        else:
            success = value == expected
        if not success:
            failures[key] = {"expected": expected, "actual": value}
    return {"question": case["question"], "passed": not failures, "failures": failures, "actual": actual}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Output already exists")
    cases = json.loads((ROOT / "evaluations/new_question_scope_cases.json").read_text(encoding="utf-8"))
    records = verified_records()
    items = [evaluate(case, records) for case in cases]
    result = {"mode": "local component benchmark; no NER model or LLM API",
              "cases": len(items), "passed": sum(i["passed"] for i in items), "items": items}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f'{result["passed"]}/{result["cases"]} component cases passed')


if __name__ == "__main__":
    main()
