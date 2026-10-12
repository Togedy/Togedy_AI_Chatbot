import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def isolate_semantic_review(monkeypatch):
    # These tests isolate deterministic citation/retrieval validation. The model
    # review protocol and its fail-closed behavior have separate tests.
    monkeypatch.setattr("llm.document_pipeline.review_document_answer", lambda raw, *a: (raw, []))

import generate_answers as answers
import search_and_export as search
from llm.evidence import NO_EVIDENCE, validate_document_answer


def response(text, page, quote):
    return json.dumps({"claims": [{"text": text, "evidence": [{"page": page, "quote": quote}]}]}, ensure_ascii=False)


def source(tmp_path, page=1, body="제출서류: 학교생활기록부와 지원자격 증빙서류", uni="연세대"):
    path = tmp_path / f"{uni}.txt"
    path.write_text(f"==== Page {page} ====\n{body}\n", encoding="utf-8")
    return {"doc_path": str(path), "page_index": page, "score": 1.0,
            "snippet": body, "matched_uni": uni, "matched_type": "수시"}


@pytest.mark.parametrize("pages,keywords", [
    (["기숙사 이용 안내", "장학금 지급 기준"], ["모집인원"]),
    (["모집 관련 문의", "기숙사 인원 안내"], ["모집인원"]),
    (["...", "   "], ["지원자격"]),
    ([], ["지원자격"]),
])
def test_rejects_unrelated_or_empty_pages(pages, keywords):
    assert search.score_pages(pages, keywords) == []


def test_single_page_and_ocr_spacing_are_searchable():
    assert search.score_pages(["제출 서류\n학교생활기록부 및 졸업증명서"], ["제출서류"])[0][0] == 0


def test_topic_synonym_matches_without_major_only_false_positive():
    pages = ["경영학과 교육과정 안내", "경영학과 구비 서류\n졸업증명서와 학교생활기록부"]
    assert [i for i, _ in search.score_pages(pages, ["경영학과", "제출서류"])] == [1]


def test_contents_page_is_not_evidence():
    pages = ["목차\n제출서류······10\n지원자격······12\n모집인원······15",
             "제출서류\n학교생활기록부 원본을 제출합니다."]
    assert [i for i, _ in search.score_pages(pages, ["제출서류"])] == [1]


def test_contents_with_leading_page_numbers_is_not_evidence():
    assert search.is_contents_page("9 주요사항\n11 제출서류\n14 모집단위 및 모집인원")


def test_dormitory_quota_is_excluded_unless_requested():
    page = "생활관(기숙사) 입사 안내\n신입생 모집인원 200명"
    assert search.score_pages([page], ["모집인원"]) == []
    assert search.score_pages([page], ["생활관", "모집인원"])


@pytest.mark.parametrize("university,keywords,excluded", [
    ("hanyang", ["모집인원"], {67}),
    ("hanyang", ["제출서류"], {64}),
    ("yonsei", ["지원자격"], {85, 86}),
    ("sogang", ["모집인원"], {10}),
])
def test_real_guides_exclude_known_irrelevant_pages(university, keywords, excluded):
    path = Path(__file__).resolve().parents[1] / "university" / university / "susi_text.txt"
    pages = search.split_text_into_labeled_pages(path.read_text(encoding="utf-8"))
    ranking = search.score_pages([page["text"] for page in pages], keywords, k=3)
    assert ranking
    assert not ({pages[index]["label"] for index, _ in ranking} & excluded)


def test_requested_page_limit_is_honored(tmp_path, monkeypatch):
    path = tmp_path / "doc.txt"
    path.write_text("\n".join(f"==== Page {i} ====\n지원자격 졸업예정자 안내 {i}" for i in range(1, 7)), encoding="utf-8")
    monkeypatch.setattr(search, "resolve_text_path", lambda *args: str(path))
    rows, _, _ = search.search_top_pages_for_query(
        "연세대 수시 지원자격", SimpleNamespace(extract_uni=lambda q: ["연세대"]),
        SimpleNamespace(extract_type=lambda q: ["수시"]),
        SimpleNamespace(extract_keywords=lambda q: []), "", "", top_pages=5,
    )
    assert len(rows) == 5
    assert all(row["page_index"] > 0 for row in rows)


@pytest.mark.parametrize("raw", [
    "자유 형식의 근거 없는 답변", "[]", '{"claims": "bad"}',
    response("모집인원은 99명입니다.", 3, "경영학과 모집인원은 20명입니다."),
    response("모집인원은 20명입니다.", 99, "경영학과 모집인원은 20명입니다."),
    response("모집인원은 20명입니다.", 3, "경영학과 모집인원은 20명으로 확정입니다."),
])
def test_rejects_invalid_claims(raw):
    assert validate_document_answer(raw, {3: "경영학과 모집인원은 20명입니다."}) == (NO_EVIDENCE, [])


def test_valid_claim_keeps_only_its_cited_page():
    quote = "경영학과 모집인원은 20명입니다."
    assert validate_document_answer(response("경영학과 모집인원은 20명입니다.", 3, quote), {3: quote, 4: "다른 자료"}) == (quote, [3])


def test_partial_invalid_claim_is_removed_with_notice():
    good = json.loads(response("경영학과 모집인원은 20명입니다.", 3, "경영학과 모집인원은 20명입니다."))
    good["claims"].append({"text": "다른 학과는 99명입니다.", "evidence": []})
    text, pages = validate_document_answer(json.dumps(good), {3: "경영학과 모집인원은 20명입니다."})
    assert "99" not in text
    assert "일부" in text
    assert pages == [3]


def test_context_rejects_error_rows_and_unreadable_pages(tmp_path):
    row = source(tmp_path)
    bad = dict(row, page_index=-1, snippet="(문서 없음)")
    missing = dict(row, doc_path=str(tmp_path / "missing.txt"))
    assert answers.build_pair_doc_context([bad, missing]) == ""


def test_quota_selection_does_not_fall_back_to_other_major(tmp_path):
    row = source(tmp_path, body="모집인원\n경제학과 30명")
    assert answers.pick_best_quota_pages([row], ["경영학과", "모집인원"]) == []


@pytest.mark.parametrize("quota", [False, True])
def test_missing_university_never_reaches_llm(tmp_path, monkeypatch, quota):
    body = "경영학과 모집인원은 20명입니다." if quota else "제출서류는 학교생활기록부입니다."
    row = source(tmp_path, body=body)
    missing = dict(row, matched_uni="고려대", page_index=-1, score=0, snippet="(문서 없음)")
    ner = {"uni": ["연세대", "고려대"], "type": ["수시"],
           "keywords": ["경영학과", "모집인원"] if quota else ["제출서류"], "decision": "문서탐색"}
    monkeypatch.setattr(answers, "search_top_pages_for_query", lambda *a, **k: ([row, missing], {}, ner))
    calls = []

    def llm(**kwargs):
        calls.append(kwargs["user_prompt"])
        assert "(문서 없음)" not in kwargs["user_prompt"]
        return response(body, 1, body)

    monkeypatch.setattr(answers, "call_llm", llm)
    monkeypatch.setattr(answers, "gpt_chat", llm)
    question = "연세대 고려대 수시 경영학과 모집인원" if quota else "연세대 고려대 수시 제출서류"
    result = answers.answer_one(question, None, None, None, "", "")
    assert len(calls) == 1
    assert body in result["answer"]
    assert NO_EVIDENCE in result["answer"]
    assert result["pair_to_rows"][("고려대", "수시")] == []


def test_invalid_generated_number_is_not_returned_or_cited(tmp_path, monkeypatch):
    row = source(tmp_path, body="경영학과 모집인원은 20명입니다.")
    ner = {"uni": ["연세대"], "type": ["수시"], "keywords": ["경영학과", "모집인원"], "decision": "문서탐색"}
    monkeypatch.setattr(answers, "search_top_pages_for_query", lambda *a, **k: ([row], {}, ner))
    monkeypatch.setattr(answers, "call_llm", lambda **k: response("경영학과는 99명입니다.", 1, row["snippet"]))
    result = answers.answer_one("연세대 수시 경영학과 모집인원", None, None, None, "", "")
    assert "99" not in result["answer"]
    assert result["pair_to_rows"][("연세대", "수시")] == []
    assert result["sources"] == []


@pytest.mark.parametrize("quota", [False, True])
def test_only_cited_pages_are_returned_as_sources(tmp_path, monkeypatch, quota):
    body = "경영학과 모집인원은 20명입니다." if quota else "제출서류는 학교생활기록부입니다."
    path = tmp_path / "guide.txt"
    path.write_text(f"==== Page 1 ====\n{body}\n==== Page 2 ====\n{body}\n", encoding="utf-8")
    rows = [dict(doc_path=str(path), page_index=i, score=1.0 / i, snippet=body,
                 matched_uni="연세대", matched_type="수시") for i in (1, 2)]
    ner = {"uni": ["연세대"], "type": ["수시"], "keywords": ["경영학과", "모집인원"] if quota else ["제출서류"], "decision": "문서탐색"}
    monkeypatch.setattr(answers, "search_top_pages_for_query", lambda *a, **k: (rows, {}, ner))
    monkeypatch.setattr(answers, "call_llm", lambda **k: response(body, 2, body))
    monkeypatch.setattr(answers, "gpt_chat", lambda **k: response(body, 2, body))
    result = answers.answer_one("연세대 수시 " + ("경영학과 모집인원" if quota else "제출서류"), None, None, None, "", "")
    assert [r["page_index"] for r in result["pair_to_rows"][("연세대", "수시")]] == [2]
    assert all("p.1" not in s for s in result["sources"])
