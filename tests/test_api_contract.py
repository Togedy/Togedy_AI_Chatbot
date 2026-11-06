import json

import pytest
import main


@pytest.mark.parametrize("body", [[], None, 1, {"first": "false"},
    {"first": 0}, {"question_1": 12}, {"question_1": None}, {"question_2": 12},
    {"NER": []}, {"NER": {"UNI": [1]}}, {"NER_Keyword": {"UNI": {}}},
    {"question_1": "가" * 10001}])
def test_invalid_input_never_calls_pipeline(monkeypatch, body):
    monkeypatch.setattr(main, "get_pipeline", lambda: pytest.fail("invalid request loaded models"))
    response = main.create_app().test_client().post("/answer", json=body)
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_malformed_json_returns_json_error():
    response = main.create_app().test_client().post("/answer", data="{", content_type="application/json")
    assert response.status_code == 400
    assert response.is_json


@pytest.mark.parametrize(("content_type", "question_2"), [
    ("application/json", ""), (None, None), ("text/plain", None),
])
def test_first_question_accepts_legacy_json_request(monkeypatch, content_type, question_2):
    monkeypatch.setattr(main, "run_turn", lambda question: (question, "답변 생성", {}, [], {}))
    response = main.create_app().test_client().post(
        "/answer",
        data=json.dumps({
            "first": True, "question_1": "첫 질문", "question_2": question_2,
            "NER_Keyword": None,
        }),
        content_type=content_type,
    )
    assert response.status_code == 200
    assert response.get_json()["answer"] == "첫 질문"


@pytest.mark.parametrize("ner", [None, {"UNI": None, "KEYWORD": ["수시"]}])
def test_followup_accepts_nullable_ner(monkeypatch, ner):
    def run_followup(previous, current, previous_ner):
        assert previous_ner == (ner or {})
        return "답변", "답변 생성", main.make_ner_payload(previous_ner), [], {}

    monkeypatch.setattr(main, "run_followup_turn", run_followup)
    response = main.create_app().test_client().post("/answer", json={
        "first": False, "question_1": "이전", "question_2": "현재", "NER_Keyword": ner,
    })
    assert response.status_code == 200
    assert response.get_json()["NER"]["UNI"] == []
    assert response.get_json()["NER"]["KEYWORD"] == (ner or {}).get("KEYWORD", [])


def test_false_uses_followup_and_preserves_response_contract(monkeypatch):
    monkeypatch.setattr(main, "run_followup_turn", lambda *a: ("답변", "답변 생성", {}, [], {}))
    client = main.create_app().test_client()
    response = client.post("/answer", json={"first": False, "question_1": "이전", "question_2": "현재"})
    assert response.status_code == 200
    assert response.get_json()["answer"] == "답변"
    response = client.post("/answer", json={"first": False, "question_1": "이전"})
    assert set(response.get_json()) == {"answer", "reply", "NER"}


def test_request_metrics_do_not_include_student_text(monkeypatch, caplog):
    monkeypatch.setattr(main, "run_turn", lambda *a: ("비공개 답변", "답변 생성", {}, [],
        {"handler": "gpt_reference", "reference_answer": {"status": "gpt_reference"},
         "coverage": [{"status": "unanswered"}], "resolved_question": "학생 개인정보"}))
    with caplog.at_level("INFO"):
        response = main.create_app().test_client().post("/answer", json={"question_1": "학생 개인정보"})
    assert response.status_code == 200
    assert "answer_metrics" in caplog.text and "gpt_reference" in caplog.text
    assert "학생 개인정보" not in caplog.text and "비공개 답변" not in caplog.text

