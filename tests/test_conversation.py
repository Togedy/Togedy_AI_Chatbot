from types import SimpleNamespace

import pytest

import generate_answers as answers
from llm.conversation import build_conversation_context, counseling_followup, student_details
from question_plan import build_question_plan


PREVIOUS = "건국대 공대 학종에서 물리 안 들었는데 괜찮을까요? 내신은 3.0이에요."
CURRENT = "화학이랑 생명을 들었고, 화학 시간에 배터리 탐구를 했어요."
NER = {"UNI": ["건국대"], "TYPE": ["수시"], "KEYWORD": ["서류평가"]}


def no_ner(monkeypatch):
    calls = []
    def extract(text, *args):
        calls.append(text)
        return {"UNI": [], "TYPE": [], "KEYWORD": []}
    monkeypatch.setattr(answers, "_extract_ner_direct", extract)
    monkeypatch.setattr(answers, "call_llm", lambda **kw: pytest.fail("No LLM needed for a profile update"))
    return calls


def test_profile_answer_uses_both_questions_without_ner_reanalysis(monkeypatch):
    calls = no_ner(monkeypatch)
    answer, decision, ner, rows, stats = answers.run_followup_turn(
        PREVIOUS, CURRENT, NER, None, None, None, "", "")
    assert calls == [CURRENT]
    assert "배터리" in answer and "3.0" in answer
    assert "어떤 과목을 이수" not in answer
    assert "지원하려는 학과" in answer
    assert answer.count("?") == 1
    assert decision == "답변 생성" and rows == []
    assert ner["UNI"] == ["건국대"]
    assert stats["handler"] == "counseling_followup"
    assert "내신" not in stats["resolved_question"]


def test_latest_grade_replaces_prior_grade():
    ctx = build_conversation_context("내신은 3.0이에요.", "정정할게요. 내신은 2.8이에요.")
    assert "2.8" in ctx.profile["내신"] and "3.0" not in ctx.profile["내신"]


def test_personal_context_can_continue_even_when_ner_is_empty(monkeypatch):
    no_ner(monkeypatch)
    answer, decision, _, rows, stats = answers.run_followup_turn(
        PREVIOUS, CURRENT, {}, None, None, None, "", "")
    assert "배터리" in answer and "3.0" in answer
    assert stats["handler"] == "counseling_followup"
    assert decision == "답변 생성" and not rows


def test_known_goal_is_not_asked_again():
    plan = build_question_plan(CURRENT)
    plan.conversation_context = build_conversation_context("화학공학과 희망해요.", CURRENT)
    answer, _ = counseling_followup(plan)
    assert "지원하려는 학과나 전공" not in answer
    assert "직접 세운 질문" in answer


@pytest.mark.parametrize("text", ["수능 60% 이상인 미대", "내신 3.0이면 가능할까요?", "친구 내신은 2.0이에요."])
def test_requirements_hypotheticals_and_others_are_not_personal_grades(text):
    assert not student_details(text)


def test_negative_activity_is_not_rewritten_as_achievement():
    ctx = build_conversation_context(PREVIOUS, "화학은 들었지만 탐구 활동은 안 했어요.")
    assert "안 했어요" in ctx.profile["활동"]


def test_negative_activity_does_not_ask_about_nonexistent_experiment():
    plan = build_question_plan("건국대 화학공학과 화학은 들었지만 탐구 활동은 안 했어요.")
    plan.conversation_context = build_conversation_context(PREVIOUS, plan.question)
    answer, _ = counseling_followup(plan)
    assert "수업에서 관심" in answer
    assert "그 활동에서 직접" not in answer


def test_asserted_grade_inside_a_question_is_preserved():
    assert "3.0" in student_details("내신은 3.0인데 지원 가능할까요?")["내신"]


def test_reset_and_separate_requests_do_not_share_profile():
    assert build_conversation_context(PREVIOUS, "다른 사람 얘기예요. 내신은 2.0이에요.").profile == {}
    assert "내신" not in build_conversation_context("", CURRENT).profile


def test_explicit_document_request_does_not_become_generic_counseling():
    plan = build_question_plan("연세대 정시 제출서류 알려줘. " + CURRENT)
    plan.conversation_context = build_conversation_context(PREVIOUS, plan.question)
    assert counseling_followup(plan) is None


def test_actual_document_followup_keeps_new_target_and_existing_sources(monkeypatch):
    no_ner(monkeypatch)
    answer, decision, ner, rows, stats = answers.run_followup_turn(
        PREVIOUS, "연세대 정시 제출서류 알려줘", NER, None, None, None, "", "")
    assert ner["UNI"] == ["연세대"] and ner["TYPE"] == ["정시"]
    assert decision == "문서탐색" and rows
    assert "기본 학생부는 별도 제출 없이" in answer
    assert "배터리" not in answer


def test_api_preserves_existing_request_and_response_fields(monkeypatch):
    import main
    no_ner(monkeypatch)
    monkeypatch.setattr(main, "get_pipeline", lambda: SimpleNamespace(
        uni_ex=None, type_ex=None, kw_ex=None, api_key="", gemini_model="", openai_model=""))
    response = main.create_app().test_client().post("/answer", json={
        "first": False, "question_1": PREVIOUS, "question_2": CURRENT, "NER": NER,
    })
    assert response.status_code == 200
    payload = response.get_json()
    assert set(payload) == {"answer", "reply", "NER"}
    assert set(payload["NER"]) == {"UNI", "TYPE", "KEYWORD"}
    assert "3.0" in payload["answer"] and "배터리" in payload["answer"]
