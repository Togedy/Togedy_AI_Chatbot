"""Re-run the answer pipeline with recorded NER and LLM outputs, without APIs.

This is a deterministic regression replay, not a fresh end-to-end evaluation.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import generate_answers as answers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Output already exists")
    recorded = json.loads(args.source.read_text(encoding="utf-8"))
    results = {"mode": "recorded NER and LLM replay; no external API", "source": str(args.source), "items": []}
    original = answers.call_llm
    try:
        for item in recorded["items"]:
            calls = list(item["llm_calls"])
            consumed = []
            def replay(**kwargs):
                if not calls:
                    raise AssertionError("Replay requested an unrecorded call")
                call = calls.pop(0)
                consumed.append(call)
                return call["output"]
            answers.call_llm = replay
            result = answers.answer_one(item["question"], None, None, None, "", "",
                                        ner_override=item["ner"])
            if calls:
                raise AssertionError(f"Unconsumed calls for item {item['id']}")
            results["items"].append({"id": item["id"], "answer": result["answer"],
                                     "changed": result["answer"] != item["answer"],
                                     "sources": result["sources"], "stats": result["stats"]})
    finally:
        answers.call_llm = original
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Replayed {len(results['items'])} items; changed IDs: " + str([i['id'] for i in results['items'] if i['changed']]))


if __name__ == "__main__":
    main()
