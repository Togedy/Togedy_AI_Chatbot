import json
import pytest
from llm.semantic_review import review_document_answer
from question_plan import build_question_plan


def run_review(response, claim_text="온라인 제공 동의자는 학생부를 별도 제출해야 합니다."):
    raw = json.dumps({"claims": [{"text": claim_text, "evidence_ids": ["p1-b0"]}]})
    bank = {"p1-b0": {"page": 1, "text": "온라인 제공 동의자는 별도 제출하지 않음"}}
    return review_document_answer(raw, bank, build_question_plan("연세대 정시 제출서류"), "제출서류", "연세대", "정시",
                                  lambda **kwargs: response, "test")


def test_negative_review_removes_claim_and_marks_unanswered():
    result, checks = run_review(json.dumps({"checks": [{"id": "c0", "supported": False, "reason": "동의 조건 반대"}]}))
    assert json.loads(result)["claims"] == []
    assert json.loads(result)["unanswered"] == ["제출서류"]


@pytest.mark.parametrize("review", ["not json", '{"checks": []}', '{"checks": [{"id":"c99","supported":true}]}',
                                    '{"checks": [{"id":"c0","supported":"true"}]}',
                                    '{"checks": [{"id":"c0","supported":true},{"id":"c0","supported":false}]}'])
def test_invalid_or_missing_approval_never_passes(review):
    result, _ = run_review(review)
    assert json.loads(result)["claims"] == []


def test_positive_review_preserves_original_claim_not_reviewers_rewrite():
    result, _ = run_review(json.dumps({"checks": [{"id": "c0", "supported": True, "evidence_id": "p1-b0", "quote": "온라인 제공 동의자는 별도 제출하지 않음", "text": "replacement"}]}), "온라인 제공 동의자는 별도 제출하지 않습니다.")
    assert json.loads(result)["claims"][0]["text"] != "replacement"

@pytest.mark.parametrize("decision", [
    {"id": "c0", "supported": True},
    {"id": "c0", "supported": True, "evidence_id": "p99-b0", "quote": "온라인 제공 동의자는 별도 제출하지 않음"},
    {"id": "c0", "supported": True, "evidence_id": "p1-b0", "quote": "원문에 없는 제출 조건입니다"},
    {"id": "c0", "supported": True, "evidence_id": "p1-b0", "quote": "동의자"},
])
def test_approved_claim_still_requires_exact_cited_support(decision):
    result, checks = run_review(json.dumps({"checks": [decision]}))
    assert json.loads(result)["claims"] == []
    assert checks[0]["supported"] is False


def test_review_rejects_evidence_ids_of_wrong_type_without_network():
    raw = json.dumps({"claims": [{"text": "주장", "evidence_ids": None}]})
    result, checks = review_document_answer(raw, {}, build_question_plan("연세대 제출서류"),
        "제출서류", "연세대", "정시", lambda **kw: pytest.fail("Invalid IDs must not call API"), "test")
    assert json.loads(result)["claims"] == []
