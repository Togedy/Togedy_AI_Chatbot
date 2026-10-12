import json
from pathlib import Path

import pytest

import generate_answers as answers
from llm.reviewed_topics import load_topics, select_topic, render_topic
from llm.evidence import evidence_blocks, validate_document_answer, NO_EVIDENCE
from question_plan import build_question_plan


def test_all_reviewed_topics_have_current_sources():
    assert len(load_topics()) == 7


def test_no_wrong_track_or_major_override():
    records = load_topics()
    assert select_topic(records, build_question_plan("연세대 정시 의예과 전형방법"), "연세대", "정시", "전형방법") is None
    assert select_topic(records, build_question_plan("건국대 수시 학생부교과 수능최저"), "건국대", "수시", "수능최저") is None


def test_changed_topic_source_invalidates_entire_record(tmp_path):
    record = load_topics()[0]
    path = tmp_path / record["sources"][0]["document"]
    path.parent.mkdir(parents=True)
    path.write_text("==== Page 17 ====\nchanged", encoding="utf-8")
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"records": [record]}), encoding="utf-8")
    assert load_topics(index, tmp_path) == []


@pytest.mark.parametrize("question,expected,forbidden", [
    ("건대 컴공 논술 최저 있어?", "2개 영역 등급 합 5", "컴퓨터공학부의 수능최저는 3개"),
    ("연세대 정시 제출서류", "기본 학생부는 별도 제출 없이", "동의 시 제출해야"),
    ("건국대 KU 일반학생 전형방법", "수능 600점(60%) + 실기 400점(40%)", "모든 모집단위는 수능 100%"),
    ("건국대 학생부종합 자유전공학부 물리 미이수 불이익", "학업역량 20% + 성장역량 50% + 공동체역량 30%", "성장역량(500점)"),
])
def test_actual_topic_answers_without_generated_table_interpretation(monkeypatch, question, expected, forbidden):
    monkeypatch.setattr(answers, "call_llm", lambda **kw: pytest.fail("Reviewed tables should not require model interpretation"))
    result = answers.answer_one(question, None, None, None, "", "")
    assert expected in result["answer"]
    assert forbidden not in result["answer"]
    assert result["sources"]


def test_mixed_schedule_keeps_both_pairs_and_sources(monkeypatch):
    monkeypatch.setattr(answers, "call_llm", lambda **kw: pytest.fail("Reviewed calendars should not require generation"))
    result = answers.answer_one("연세대 정시랑 고려대 수시 경영학과 일정", None, None, None, "", "")
    assert "2025.12.29." in result["answer"] and "2025.09.08." in result["answer"]
    assert set(result["pair_to_rows"]) == {("연세대", "정시"), ("고려대", "수시")}


def test_mixed_evaluation_columns_cannot_form_false_total():
    claim = "평가요소는 학업역량(300점), 진로역량(400점), 성장역량(500점), 공동체역량(300점)으로 총 1,000점입니다."
    pages = {1: claim}
    raw = json.dumps({"claims": [{"text": claim, "evidence_ids": ["p1-b0"]}]})
    assert validate_document_answer(raw, pages, blocks=evidence_blocks(pages)) == (NO_EVIDENCE, [])
