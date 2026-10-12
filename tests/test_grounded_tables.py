import json
from pathlib import Path

import pytest

import generate_answers as answers
from llm.evidence import evidence_blocks, validate_document_answer
from llm.table_evidence import english_grade_answer
from llm.reviewed_facts import verified_records, matches, discovery_answer
from llm.major_resolution import major_name_clarification
from question_plan import build_question_plan

ROOT = Path(__file__).resolve().parents[1]


def test_complete_page_keeps_late_english_table():
    page = answers.load_page_text(str(ROOT / "university/konkuk/jungsi_text.txt"), 35)
    blocks = evidence_blocks({35: page})
    assert blocks["p35-b0"]["text"] == page
    raw = json.dumps({"claims": [{"text": "영어 2등급은 197점입니다.", "evidence_ids": ["p35-b0"]}]})
    assert validate_document_answer(raw, {35: page}, blocks=blocks)[1] == [35]


def test_english_table_has_nine_columns_in_original_order():
    page = answers.load_page_text(str(ROOT / "university/konkuk/jungsi_text.txt"), 35)
    answer, pages = english_grade_answer({35: page})
    assert "1등급 200점" in answer and "9등급 150점" in answer
    assert answer.count("등급 ") == 9
    assert pages == [35]
    assert english_grade_answer({35: page.replace("197", "missing")}) == ("", [])


def test_reviewed_records_match_current_sources():
    assert len(verified_records()) == 3


def test_changed_source_invalidates_reviewed_record(tmp_path):
    record = verified_records()[0]
    path = tmp_path / record["document"]
    path.parent.mkdir(parents=True)
    path.write_text("==== Page 24 ====\nchanged", encoding="utf-8")
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"facts": [record]}), encoding="utf-8")
    assert verified_records(index, tmp_path) == []


@pytest.mark.parametrize("question,expected,forbidden", [
    ("실기 없는 미대 전형", "서울대", "한양대"),
    ("실기 전형 60%이상인 미대", "한양대", "건국대"),
    ("수능 전형 60%이상인 미대", "건국대", "한양대"),
])
def test_art_constraints_are_applied_to_reviewed_final_stage(question, expected, forbidden):
    result = discovery_answer(build_question_plan(question))
    assert expected in result[0] and forbidden not in result[0]
    assert result[1]


def test_high_first_stage_does_not_satisfy_final_stage_constraint():
    record = dict(verified_records()[1])
    record["stages"] = [{"components": {"실기": 100}}, {"components": {"실기": 40, "수능": 60}}]
    assert not matches(record, build_question_plan("실기 전형 60%이상 미대"))


def test_university_filter_does_not_return_other_university_example():
    assert discovery_answer(build_question_plan("건국대 실기 없는 미대")) is None


def test_unknown_major_gets_real_names_instead_of_silent_alias():
    answer = major_name_clarification(build_question_plan("건국대 자유전공공학부 들어가는 방법"))
    assert answer and "공과대학자유전공학부" in answer
    assert major_name_clarification(build_question_plan("건국대 컴퓨터공학부 들어가는 방법")) is None


def test_compound_request_generates_each_topic_separately(monkeypatch, tmp_path):
    monkeypatch.setattr("llm.document_pipeline.review_document_answer", lambda raw, *a: (raw, []))
    path = tmp_path / "guide.txt"
    text = "일반전형 인문계 전형방법은 수능 100%입니다. 제출서류는 학교생활기록부입니다."
    path.write_text("==== Page 1 ====\n" + text, encoding="utf-8")
    row = dict(doc_path=str(path), page_index=1, matched_uni="연세대", matched_type="정시", score=1)
    monkeypatch.setattr(answers, "search_top_pages_for_query", lambda *a, **k: ([row], {}, {}))
    prompts = []
    def llm(**kwargs):
        prompts.append(kwargs["user_prompt"])
        claim = "제출서류는 학교생활기록부입니다." if "이번에 답할 항목: 제출서류" in prompts[-1] else "일반전형 인문계 전형방법은 수능 100%입니다."
        return json.dumps({"claims": [{"text": claim, "evidence_ids": ["p1-b0"]}]})
    monkeypatch.setattr(answers, "call_llm", llm)
    result = answers.answer_one("연세대 정시 전형방법과 제출서류", None, None, None, "", "")
    assert len(prompts) == 2
    assert len(result["stats"]["coverage"]) == 2
    assert "[제출서류]" in result["answer"] and "[전형방법]" in result["answer"]


def test_major_correction_keeps_requested_method():
    ner = answers.merge_ner_context(
        {"UNI": ["건국대"], "KEYWORD": ["자유전공공학부", "전형방법"]},
        {"KEYWORD": ["공과대학자유전공학부"]})
    assert ner["KEYWORD"] == ["공과대학자유전공학부", "전형방법"]


def test_topic_followup_keeps_selected_major():
    ner = answers.merge_ner_context(
        {"UNI": ["건국대"], "KEYWORD": ["컴퓨터공학부", "전형방법"]},
        {"KEYWORD": ["제출서류"]})
    assert ner["KEYWORD"] == ["컴퓨터공학부", "제출서류"]
