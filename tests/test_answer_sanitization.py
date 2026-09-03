from generate_answers import (
    apply_explicit_followup_rules,
    build_pair_doc_context,
    sanitize_pair_answer,
)
from search_and_export import extract_snippet_around_keywords


def test_pair_answer_preserves_paragraphs_and_lists():
    answer = (
        "2026학년도 건국대학교 수시 전형 일정은 다음과 같습니다.\n\n"
        "- 원서접수: 2025년 9월 8일~11일\n"
        "- 합격자 발표: 2025년 12월 12일"
    )

    assert sanitize_pair_answer(answer) == answer


def test_pair_answer_removes_only_followup_section():
    answer = (
        "연세대학교 정시 제출 서류는 다음과 같습니다.\n\n"
        "- 학교생활기록부\n"
        "- 지원자격 증빙서류\n\n"
        "추가로 다른 전형도 알려드릴까요?"
    )

    assert sanitize_pair_answer(answer) == (
        "연세대학교 정시 제출 서류는 다음과 같습니다.\n\n"
        "- 학교생활기록부\n"
        "- 지원자격 증빙서류"
    )


def test_explicit_rules_preserve_every_requested_keyword():
    ner = {"UNI": ["연세대"], "TYPE": ["정시"], "KEYWORD": []}

    result = apply_explicit_followup_rules(
        "연세대 정시 전형방법과 제출 서류 알려줘",
        ner,
    )

    assert result["KEYWORD"] == ["제출서류", "전형방법"]


def test_search_snippet_preserves_table_line_breaks():
    page = "항목\t일정\n원서접수\t9월 8일\n서류제출\t9월 12일"

    snippet = extract_snippet_around_keywords(page, ["원서접수"], window=200)

    assert snippet == page


def test_pair_context_prefers_full_page_over_short_snippet(tmp_path):
    document = tmp_path / "susi_text.txt"
    document.write_text(
        "==== Page 13 ====\n"
        "전형 일정\n"
        "항목\t기간\n"
        "원서접수\t2025년 9월 8일~11일\n"
        "서류제출\t2025년 9월 12일까지\n",
        encoding="utf-8",
    )
    rows = [{
        "doc_path": str(document),
        "page_index": 13,
        "score": 1.0,
        "snippet": "전형 일정",
    }]

    context = build_pair_doc_context(rows)

    assert "항목\t기간" in context
    assert "원서접수\t2025년 9월 8일~11일" in context
    assert "서류제출\t2025년 9월 12일까지" in context
