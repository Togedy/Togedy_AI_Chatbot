import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import generate_answers as answers
from question_plan import build_question_plan, local_response
from query_constraints import parse_percentages, invalid_conditions
from llm.reviewed_facts import verified_records, matches, discovery_answer
from llm.evidence import evidence_blocks, validate_document_answer, NO_EVIDENCE
from evaluations.evaluate_new_question_scope import evaluate
from search_and_export import search_top_pages_for_query

ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads((ROOT / "evaluations/new_question_scope_cases.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", CASES, ids=[str(n) for n in range(1, len(CASES) + 1)])
def test_frozen_new_question_scopes(case):
    result = evaluate(case, verified_records())
    assert result["passed"], result["failures"]


@pytest.mark.parametrize("question", ["실기 120% 이상 미대", "실기 80% 이상 실기 50% 이하 미대", "60% 이상인 미대"])
def test_invalid_constraints_do_not_call_search_or_model(monkeypatch, question):
    monkeypatch.setattr(answers, "search_top_pages_for_query", lambda *a, **k: pytest.fail("Invalid conditions must be clarified first"))
    monkeypatch.setattr(answers, "call_llm", lambda **k: pytest.fail("No ungrounded answer"))
    result = answers.answer_one(question, None, None, None, "", "")
    assert result["decision"] == "재질문"
    assert result["sources"] == []


def test_excluded_university_never_reaches_document_search(monkeypatch):
    universities = []
    def resolve(uni, round_name):
        universities.append(uni)
        return str(ROOT / "missing-test-guide.txt")
    monkeypatch.setattr("search_and_export.resolve_text_path", resolve)
    plan = build_question_plan("건국대 제외하고 수능 60% 이상인 미대")
    search_top_pages_for_query(plan.question, None, None, None, "", "", query_plan=plan)
    assert universities and "konkuk" not in universities


@pytest.mark.parametrize("claim,question", [
    ("현대미술학과는 수능 60% + 실기 40%입니다.", "수능 60% 초과 미대"),
    ("현대미술학과는 수능 60% + 실기 40%입니다.", "수능 60% 이상 실기 30% 이하 미대"),
])
def test_generated_claim_cannot_evade_maximum_or_strict_bound(claim, question):
    pages = {18: claim}
    raw = json.dumps({"claims": [{"text": claim, "evidence_ids": ["p18-b0"]}]})
    plan = build_question_plan(question)
    assert validate_document_answer(raw, pages, blocks=evidence_blocks(pages), constraints=plan.constraints) == (NO_EVIDENCE, [])


def test_followup_named_track_changes_round_and_keeps_topic(monkeypatch):
    uni = SimpleNamespace(extract_uni=lambda t: [])
    typ = SimpleNamespace(extract_type=lambda t: [])
    kw = SimpleNamespace(extract_keywords=lambda t: ["KU일반학생"])
    captured = []
    def answer(text, *a, **k):
        captured.append((text, k["ner_override"]))
        ner = k["ner_override"]
        return dict(answer="confirmed", decision="문서탐색", ner_uni=ner["UNI"],
                    ner_type=ner["TYPE"], ner_kw=ner["KEYWORD"], pair_to_rows={}, stats={})
    monkeypatch.setattr(answers, "answer_one", answer)
    result = answers.run_followup_turn("건국대 수시 제출서류", "KU일반학생은?",
        {"UNI": ["건국대"], "TYPE": ["수시"], "KEYWORD": ["제출서류"]}, uni, typ, kw, "", "")
    assert captured[0][1]["TYPE"] == ["정시"]
    assert "제출서류" in captured[0][1]["KEYWORD"]
    assert "KU일반학생" in captured[0][0]
    assert result[2]["TYPE"] == ["정시"]


def test_followup_preparatory_major_is_preserved_when_topic_changes():
    ner = answers.merge_ner_context(
        {"UNI": ["연세대"], "TYPE": ["정시"], "KEYWORD": ["의예과", "전형방법"]},
        {"KEYWORD": ["제출서류"]})
    assert ner["KEYWORD"] == ["의예과", "제출서류"]


def test_compound_discovery_does_not_silently_drop_quota():
    assert discovery_answer(build_question_plan("서울대 미대 실기 없이 모집인원")) is None

def test_explicit_first_stage_does_not_use_final_stage_ratio():
    records = verified_records()
    plan = build_question_plan("한양대 미대 1단계 실기 100%")
    assert not any(matches(r, plan) for r in records)
    plan = build_question_plan("한양대 미대 2단계 실기 100%")
    assert [r["id"] for r in records if matches(r, plan)] == ["hanyang-susi-art-talent"]


def test_multiple_stages_with_unassigned_percentages_get_clarification():
    plan = build_question_plan("미대 1단계와 2단계 실기 60% 이상")
    assert plan.intent == "clarify_constraints"
    assert "단계를 구분" in local_response(plan)[0]

def test_real_calendar_cells_are_extracted_with_explicit_scope():
    from llm.table_evidence import calendar_table_answer
    path = ROOT / "university/konkuk/susi_text.txt"
    pages = {19: answers.load_page_text(str(path), 19)}
    answer, cited = calendar_table_answer(str(path), pages)
    assert "원서접수 · 전체 전형" in answer
    assert "2025. 9. 9.(화) 10:00" in answer
    assert cited == [19]
    assert calendar_table_answer(str(path), pages, ["논술"]) == ("", [])


def test_date_in_csv_must_exist_on_the_original_page(tmp_path):
    from llm.table_evidence import calendar_table_answer
    path = tmp_path / "susi_text.txt"
    path.write_text("", encoding="utf-8")
    table = tmp_path / "susi_tables.csv"
    table.write_text("Page,0,1\n1,구분,일정\n1,원서접수,2025.9.9.\n1,서류제출,2025.9.16.\n", encoding="utf-8")
    assert calendar_table_answer(str(path), {1: "원서접수 2025.9.9. 서류제출 2025.9.17."}) == ("", [])
    answer, cited = calendar_table_answer(str(path), {1: "원서접수 2025.9.9. 서류제출 2025.9.16."})
    assert cited == [1]


def test_multiple_date_columns_and_unlabelled_cells_are_not_guessed(tmp_path):
    from llm.table_evidence import calendar_table_answer
    path = tmp_path / "susi_text.txt"
    table = tmp_path / "susi_tables.csv"
    table.write_text("Page,0,1,2\n1,구분,일정,일정\n1,원서접수,2025.9.9.,2025.9.10.\n1,,2025.9.16.,\n", encoding="utf-8")
    assert calendar_table_answer(str(path), {1: "원서접수 2025.9.9. 2025.9.10. 2025.9.16."}) == ("", [])

def test_csv_cannot_swap_two_dates_that_both_exist_on_same_page(tmp_path):
    from llm.table_evidence import calendar_table_answer
    path = tmp_path / "susi_text.txt"
    table = tmp_path / "susi_tables.csv"
    table.write_text("Page,0,1\n1,구분,일정\n1,원서접수,2025.9.16.\n1,서류제출,2025.9.9.\n", encoding="utf-8")
    assert calendar_table_answer(str(path), {1: "원서접수 2025.9.9. 서류제출 2025.9.16."}) == ("", [])

def test_common_calendar_is_preferred_to_one_specific_track():
    from llm.table_evidence import calendar_table_answer
    path = ROOT / "university/konkuk/susi_text.txt"
    pages = {24: answers.load_page_text(str(path), 24), 19: answers.load_page_text(str(path), 19)}
    answer, cited = calendar_table_answer(str(path), pages)
    assert cited == [19]
    assert "전체 전형" in answer

def test_track_name_alone_is_not_used_as_an_event_label(tmp_path):
    from llm.table_evidence import calendar_table_answer
    path = tmp_path / "susi_text.txt"
    (tmp_path / "susi_tables.csv").write_text(
        "Page,0,1\n1,구분,일정\n1,원서접수,2025.9.9.\n1,학생부종합(KU자기추천),2025.11.14.\n", encoding="utf-8")
    assert calendar_table_answer(str(path), {1: "원서접수 2025.9.9. 학생부종합(KU자기추천) 2025.11.14."}) == ("", [])
