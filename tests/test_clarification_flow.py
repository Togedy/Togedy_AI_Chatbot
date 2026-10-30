from types import SimpleNamespace

import pytest

import generate_answers as answers
import main


@pytest.fixture
def client(monkeypatch):
    # Keep actual routing, slot merging, retrieval, reviewed data and source
    # metadata. Replace only the heavy NER models; no external generation.
    monkeypatch.setattr(answers, "_extract_ner_direct", lambda *a: {"UNI": [], "TYPE": [], "KEYWORD": []})
    monkeypatch.setattr(answers, "call_llm", lambda **kw: pytest.fail("Unexpected model call"))
    monkeypatch.setattr(main, "get_pipeline", lambda: SimpleNamespace(
        uni_ex=None, type_ex=None, kw_ex=None, api_key="", gemini_model="", openai_model=""))
    return main.create_app().test_client()


def ask(client, question, previous=None, ner=None):
    body = {"first": previous is None, "question_1": previous or question}
    if previous is not None:
        body.update(question_2=question, NER=ner or {})
    response = client.post("/answer", json=body)
    assert response.status_code == 200
    return response.get_json()


def test_missing_university_then_filled_uses_actual_correct_document(client):
    first = ask(client, "정시 제출서류 알려줘")
    assert first["reply"] is True and "대학" in first["answer"]
    assert "location" not in first
    second = ask(client, "연세대", "정시 제출서류 알려줘", first["NER"])
    assert second["reply"] is False
    assert second["NER"]["UNI"] == ["연세대"] and second["NER"]["TYPE"] == ["정시"]
    assert second["location"].replace("\\", "/") == "university/yonsei/jungsi.pdf"
    assert second["NER_Page_1"].startswith("p")
    assert "기본 학생부는 별도 제출 없이" in second["answer"]


def test_partial_followup_reasks_full_form_without_search(client, monkeypatch):
    monkeypatch.setattr(answers, "search_top_pages_for_query", lambda *a, **k: pytest.fail("Incomplete target must not search"))
    first = ask(client, "제출서류 알려줘")
    second = ask(client, "연세대", "제출서류 알려줘", first["NER"])
    assert second["reply"] is True
    for field in ("대학: 연세대", "전형(수시·정시", "학과·모집단위", "궁금한 내용", "한 메시지"):
        assert field in second["answer"]
    assert "제출서류" in second["NER"]["KEYWORD"]
    assert "location" not in second and not any(k.startswith("NER_Page_") for k in second)


def test_full_answer_to_retry_form_resumes_document_lookup(client):
    first = ask(client, "제출서류 알려줘")
    second = ask(client, "연세대", "제출서류 알려줘", first["NER"])
    third = ask(client, "대학: 연세대\n전형: 정시\n학과·모집단위: 전체\n궁금한 내용: 제출서류", "연세대", second["NER"])
    assert third["reply"] is False
    assert third["location"].replace("\\", "/") == "university/yonsei/jungsi.pdf"


@pytest.mark.parametrize("previous,current", [
    ("연세대 제출서류 알려줘", "정시요"),
    ("연세대 정시", "제출서류요"),
])
def test_filling_round_or_topic_also_reaches_correct_document(client, previous, current):
    first = ask(client, previous)
    assert first["reply"] is True
    second = ask(client, current, previous, first["NER"])
    assert second["reply"] is False
    assert second["location"].replace("\\", "/") == "university/yonsei/jungsi.pdf"
    assert "기본 학생부는 별도 제출 없이" in second["answer"]


def test_ner_university_also_emitted_as_keyword_does_not_erase_original_topic(client, monkeypatch):
    first = ask(client, "정시 제출서류 알려줘")
    monkeypatch.setattr(answers, "_extract_ner_direct", lambda *a: {
        "UNI": ["연세대"], "TYPE": [], "KEYWORD": ["연세대"],
    })
    second = ask(client, "연세대", "정시 제출서류 알려줘", first["NER"])
    assert second["reply"] is False
    assert "제출서류" in second["NER"]["KEYWORD"]
    assert second["location"].replace("\\", "/") == "university/yonsei/jungsi.pdf"


@pytest.mark.parametrize("text", ["네", "잘 모르겠어요", "아무거나요"])
def test_nonanswer_to_missing_university_returns_full_retry(client, text, monkeypatch):
    monkeypatch.setattr(answers, "search_top_pages_for_query", lambda *a, **k: pytest.fail("Must clarify"))
    first = ask(client, "정시 제출서류 알려줘")
    second = ask(client, text, "정시 제출서류 알려줘", first["NER"])
    assert second["reply"] is True and "한 메시지" in second["answer"]
    assert "전형: 정시" in second["answer"] and "제출서류" in second["answer"]


def test_partial_ner_still_recovers_previous_requested_topic(client):
    second = ask(client, "연세대", "정시 제출서류 알려줘", {"TYPE": ["정시"]})
    assert second["reply"] is False
    assert second["location"].replace("\\", "/") == "university/yonsei/jungsi.pdf"


def test_unfilled_form_options_are_not_treated_as_user_choices(client, monkeypatch):
    monkeypatch.setattr(answers, "search_top_pages_for_query", lambda *a, **k: pytest.fail("Round was not filled"))
    response = ask(client,
        "대학: 연세대\n전형(수시·정시, 알면 세부 전형명도): \n"
        "학과·모집단위(전체 안내를 원하면 '전체'): 전체\n"
        "궁금한 내용(예: 제출서류·일정·모집인원): 제출서류",
        "제출서류 알려줘", {"KEYWORD": ["제출서류"]})
    assert response["reply"] is True
    assert response["NER"]["TYPE"] == []
    assert "location" not in response


def test_unanswered_counseling_question_reasks_whole_context(client):
    previous = "화학이랑 생명을 들었고, 배터리 탐구를 했어요. 내신은 3.0이에요."
    ner = {"UNI": ["건국대"], "TYPE": ["수시"], "KEYWORD": ["서류평가"]}
    response = ask(client, "잘 모르겠어요", previous, ner)
    assert response["reply"] is True
    for text in ("희망 학과·전공", "현재 성적", "3.0", "배터리", "궁금한 점", "한 메시지"):
        assert text in response["answer"]
    assert "location" not in response


def test_other_personal_detail_does_not_fill_missing_major(client):
    previous = "배터리 탐구를 했어요."
    response = ask(client, "내신은 2.8이에요.", previous,
                   {"UNI": ["건국대"], "TYPE": ["수시"], "KEYWORD": ["서류평가"]})
    assert response["reply"] is True and "희망 학과·전공" in response["answer"]
    assert "2.8" in response["answer"]
