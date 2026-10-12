"""Run the user's 23 image questions through the unchanged production pipeline.

Run from the repository root with Python 3.11. Outputs are local evaluation
artifacts; API keys and request headers are never recorded.
"""
import json
import os
import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

QUESTIONS = [
    "건대 수시 전형 일정 알려줘",
    "연세대 정시 전형 방법과 제출서류",
    "영어등급 환산표와 KU 일반학생 전형 안내",
    "건대랑 서연고 수시 전형 일정 알려줘",
    "나 서울대 정시랑 수시에서 컴퓨터공학부 몇명 뽑는지 궁금해",
    "부산 사는 사람인데 건대가 메리트 있을까?",
    "연세대랑 고려대중에 어디가 더 좋아?",
    "연고대중에서 학교를 어디를 써야 좋을지 고민중이야",
    "경영학과를 쓰려고 하는데 연세대 정시랑 고려대 수시 둘중에 어디가 붙기 쉬울까?\n연세대 정시랑 고려대 수시 각각의 전형일정이 궁금해",
    "연세대랑 고려대 중에서 정시로 경영학과를 고민중에 있는데 어디가 더 잘 붙을까?",
    "건대 컴공과 논술 최저가 있는 거야? 없는거야?",
    "작년 서성한중경외시 중에 인원 미달인 학교를 알려줘",
    "실기 없는 미대 입시 전형 정리해줘",
    "건대 수시랑 연대 수시 중에 서양회과 예비 번호는 어디가 더 잘빠져?",
    "건국대 학생부 종합전형을 쓰려고 하는데 컴퓨터공학부말고 또 경쟁률이 6:1이랑 비슷한 곳이 있나요?",
    "내가 공부를 못해서 실기 전형 60%이상인 미대 전형을 찾아줘",
    "내가 실기를 못해서 수능 전형 60%이상인 미대 전형을 찾아줘",
    "건국대 자유전공공학부 들어가는 방법에 대해 알려줘",
    "타학교 산업공학과 목표로 물리를 안들었는데 건국대 학생부종합 지원하려면 불이익이 클까요? 대부분 공대는 물리 안들으면 학생부 종합 전형 광탈이라고 하니까요... 내신도 3.0이예요 일반고 생기부 열심히 챙기긴 했는데 힘들까요? 자유전공학부는 어떤가요?",
    "건대 경영 가려면 정시 백분위 몇 정도 맞아야 되는지 궁금합니다",
    "건국대 경영관 위치나 시설 좋은가요?",
    "현대미술과 희망하는데 학생부 종합전형에서 기타과목 중요도가 궁금합니다",
    "나 심심해",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", help="New result filename; existing results are never overwritten")
    parser.add_argument("--ner-from", type=Path, help="Reuse saved NER for the same questions; retrieval and LLM calls remain live")
    args = parser.parse_args()
    offline = args.offline
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import main as app
    import generate_answers as answers
    import search_and_export as search
    from settings import Settings

    directory = Path(__file__).resolve().parent
    output = directory / (args.output or ("image_questions_offline.json" if offline else "image_questions_results.json"))
    if output.exists():
        raise SystemExit("Result already exists. Choose a new --output filename.")
    config = Settings.from_env()
    results = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "openai_model": config.openai_model,
        "gemini_model": config.gemini_model,
        "mode": "independent first turns; item 9 is one combined question",
        "offline": offline,
        "excluded_from_primary_score": [12],
        "items": [],
    }
    saved_ner = {}
    if args.ner_from:
        if offline:
            raise SystemExit("--ner-from is only supported for live answer evaluation")
        saved = json.loads(args.ner_from.read_text(encoding="utf-8"))
        saved_ner = {item["question"]: item["ner"] for item in saved["items"]}
        if any(q not in saved_ner for q in QUESTIONS):
            raise SystemExit("Saved NER must contain every exact question")
        results["mode"] = "saved NER; live retrieval, generation and semantic verification; independent turns"
        results["ner_source"] = str(args.ner_from)
        pipeline = None
    else:
        print("Loading the existing NER pipeline...", flush=True)
        pipeline = app.get_pipeline()
    if offline:
        # Offline retrieval assessment only: no LLM answer or Gemini grouping.
        search.gemini_sort = lambda *args, **kwargs: []
    original = answers.call_llm
    calls = []

    def traced_call(*args, **kwargs):
        started = time.perf_counter()
        result = original(*args, **kwargs)
        calls.append({"output": result, "elapsed_seconds": round(time.perf_counter() - started, 3)})
        return result

    answers.call_llm = traced_call
    for number, question in enumerate(QUESTIONS, 1):
        calls.clear()
        started = time.perf_counter()
        item = {"id": number, "question": question, "in_scope": number != 12}
        try:
            if offline:
                rows, stats, ner = search.search_top_pages_for_query(
                    question, pipeline.uni_ex, pipeline.type_ex, pipeline.kw_ex, "", "", top_pages=3)
                item.update(ner=ner, rows=rows, stats=stats, decision=ner.get("decision"))
            else:
                if saved_ner:
                    result = answers.answer_one(question, None, None, None, config.google_api_key, config.gemini_model,
                                                llm_model=config.openai_model, ner_override=saved_ner[question])
                    answer, decision = result["answer"], result["decision"]
                    ner = {"UNI": result["ner_uni"], "TYPE": result["ner_type"], "KEYWORD": result["ner_kw"]}
                    rows, stats = answers._flatten_rows(result["pair_to_rows"]), result["stats"]
                else:
                    answer, decision, ner, rows, stats = app.run_turn(question)
                item.update(answer=answer, decision=decision, ner=ner, rows=rows,
                            stats=stats, api_response=app.build_answer_response(answer, decision, rows, ner))
        except Exception as exc:
            item.update(error_type=type(exc).__name__)
        item["llm_calls"] = list(calls)
        item["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        results["items"].append(item)
        output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({key: item.get(key) for key in ("id", "ner", "decision", "answer", "error_type", "elapsed_seconds")}, ensure_ascii=False), flush=True)
    print(f"Saved {len(results['items'])} results to {output}", flush=True)


if __name__ == "__main__":
    main()
