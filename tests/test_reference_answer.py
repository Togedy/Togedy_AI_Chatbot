import json

import pytest
import generate_answers as answers
from llm.reference_answer import REFERENCE_NOTICE, reference_answer
from llm.calendar_evidence import calendar_text_answer, calendar_claim_matches
from question_plan import build_question_plan


SCHEDULE = """입학전형 일정표
학생부종합전형(지역균형전형)
지원서 접수(인터넷 접수)
2025. 9. 8.(월) 10:00 ~ 9. 10.(수) 18:00
서류제출(해당자)
2025. 9. 8.(월) ~ 9. 11.(목) 10:00 ~ 18:00
합격자 발표
2025. 12. 12.(금)
"""


def test_schedule_preserves_closing_time_and_track_scope():
    text, pages = calendar_text_answer({5: SCHEDULE})
    assert pages == [5]
    assert "9. 11.(목) 10:00 ~ 18:00" in text
    assert calendar_text_answer({5: SCHEDULE}, ["일반전형"]) == ("", [])
    assert not calendar_claim_matches("서류제출은 9월 11일 10:00까지입니다.", SCHEDULE)
    assert calendar_claim_matches("서류제출은 9월 11일 18:00까지입니다.", SCHEDULE)


@pytest.mark.parametrize("question", ["건대 경영 정시 백분위 몇 정도?", "건국대 경영관 시설 좋은가요?"])
def test_reference_notice_is_enforced_without_document_sources(monkeypatch, question):
    monkeypatch.setattr(answers, "call_llm", lambda **kw: json.dumps({"answer": "지원 연도와 전형에 따라 달라요. 비교 기준부터 살펴보세요."}))
    result = answers.answer_one(question, None, None, None, "", "")
    assert REFERENCE_NOTICE in result["answer"]
    assert result["stats"]["reference_answer"]["verified"] is False
    assert result["stats"]["handler"] == "gpt_reference"
    assert result["sources"] == []
    assert not any(result["pair_to_rows"].values())


def test_missing_documents_use_separate_reference_prompt(monkeypatch):
    monkeypatch.setattr(answers, "search_top_pages_for_query", lambda *a, **kw: ([], {}, {}))
    prompts = []
    def call(**kw):
        prompts.append(kw)
        return json.dumps({"answer": "전형의 지원 자격과 평가 요소를 함께 확인하세요."})
    monkeypatch.setattr(answers, "call_llm", call)
    result = answers.answer_one("건국대 수시 전형방법", None, None, None, "", "")
    assert len(prompts) == 1
    assert "<source_blocks>" not in prompts[0]["user_prompt"]
    assert REFERENCE_NOTICE in result["answer"]
    assert result["sources"] == []


@pytest.mark.parametrize("raw", ['{"claims": []}', 'not json', '{"answer": ""}'])
def test_invalid_reference_does_not_become_an_answer(raw):
    text, status = reference_answer(build_question_plan("건대 경영 정시 백분위"), lambda **kw: raw, "test")
    assert text == ""
    assert status["status"] == "unavailable"


def test_major_correction_preserves_specific_topic():
    previous = {"UNI": ["건국대"], "TYPE": ["수시"], "KEYWORD": ["현대미술과", "기타과목", "학생부종합", "서류평가"]}
    current = {"UNI": [], "TYPE": [], "KEYWORD": ["현대미술학과"]}
    merged = answers.merge_ner_context(previous, current)
    assert "기타과목" in merged["KEYWORD"]
    assert "학생부종합" in merged["KEYWORD"]


@pytest.mark.parametrize("first", ["해당 사항 없음", "합격 백분위는 80% 이상을 목표로 해보세요."])
def test_reference_retries_empty_or_speculative_numeric_advice(first):
    replies = iter([first, "대학별 환산 방식이 달라요. 전년도 결과의 기준과 자신의 환산점수를 함께 비교하세요."])
    calls = []
    def call(**kw):
        calls.append(kw)
        return json.dumps({"answer": next(replies)})
    text, status = reference_answer(build_question_plan("건대 경영 정시 백분위"), call, "test")
    assert len(calls) == 2
    assert first not in text
    assert REFERENCE_NOTICE in text
    assert status["status"] == "gpt_reference"


def test_incorrect_closing_time_is_rejected_by_document_gate():
    from llm.evidence import validate_document_answer, NO_EVIDENCE
    raw = json.dumps({"claims": [{"text": "서류제출은 9월 11일 10:00까지입니다.",
                      "evidence": [{"page": 5, "quote": SCHEDULE}]}]})
    answer, pages = validate_document_answer(raw, {5: SCHEDULE})
    assert answer == NO_EVIDENCE
    assert pages == []


def test_reference_discards_speculative_requirements_but_keeps_gpt_advice():
    raw = json.dumps({"answer": "자기소개서와 추천서를 제출해야 해요. 전형별 지원 자격과 교과 반영 방식을 공식 자료에서 함께 확인해 보세요."})
    text, status = reference_answer(build_question_plan("건국대 수시 전형방법"), lambda **kw: raw, "test")
    assert "자기소개서" not in text
    assert "추천서" not in text
    assert "교과 반영 방식" in text
    assert status["status"] == "gpt_reference"


def test_college_only_reply_keeps_specific_question_and_track():
    merged = answers.merge_ner_context(
        {"UNI": [], "TYPE": ["수시"], "KEYWORD": ["현대미술과", "기타과목", "학생부종합", "서류평가"]},
        {"UNI": ["건국대"], "TYPE": [], "KEYWORD": []})
    assert "기타과목" in merged["KEYWORD"]
    assert "학생부종합" in merged["KEYWORD"]


def test_major_method_needs_specific_track_before_answering(monkeypatch):
    monkeypatch.setattr("llm.document_pipeline.major_name_clarification", lambda plan: None)
    result = answers.answer_one("건국대 수시 공과대학자유전공학부 전형방법", None, None, None, "", "")
    assert result["decision"] == "재질문"
    assert result["stats"]["handler"] == "track_clarification"


def test_unanswered_specific_subject_gets_reference_even_if_ner_omits_it(monkeypatch):
    monkeypatch.setattr("llm.document_pipeline.local_response", lambda plan: ("기타과목의 중요성은 평가 요소에 따라 달라요.", "답변 생성"))
    monkeypatch.setattr(answers, "call_llm", lambda **kw: json.dumps({"answer": "기타과목은 성적만 보기보다 수강 맥락과 학습 활동을 함께 살펴보세요."}))
    result = answers.answer_one("건국대 수시 기타과목의 중요도가 궁금해", None, None, None, "", "",
                                ner_override={"UNI": ["건국대"], "TYPE": ["수시"], "KEYWORD": ["서류평가"]})
    assert REFERENCE_NOTICE in result["answer"]
