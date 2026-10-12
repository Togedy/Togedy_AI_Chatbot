import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import generate_answers as answers
from question_plan import build_question_plan, local_response
from search_and_export import search_top_pages_for_query, build_pairs_smart
from llm.evidence import _numbers, evidence_blocks, validate_document_answer, NO_EVIDENCE
from llm.table_evidence import quota_answer


def test_explicit_rules_preserve_rounds_and_major():
    ner = answers.apply_explicit_followup_rules(
        "서울대 수시랑 정시에서 컴퓨터공학부 몇명 뽑아",
        {"UNI": ["서울대"], "TYPE": ["수시", "정시"], "KEYWORD": ["컴퓨터공학부"]},
    )
    assert ner["TYPE"] == ["수시", "정시"]
    assert ner["KEYWORD"] == ["컴퓨터공학부", "모집인원"]


@pytest.mark.parametrize("question,universities,rounds,topic", [
    ("건대 컴공 논술 최저가 있어?", ["건국대"], ["수시"], "수능최저"),
    ("영어등급 환산표와 KU 일반학생 전형 안내", ["건국대"], ["정시"], "영어등급"),
    ("연세대 정시 전형방법과 제출서류", ["연세대"], ["정시"], "제출서류"),
])
def test_explicit_track_and_topic_planning(question, universities, rounds, topic):
    plan = build_question_plan(question)
    assert plan.universities == universities
    assert plan.rounds == rounds
    assert topic in plan.topics
    assert plan.intent == "fact"


def test_background_major_does_not_become_target():
    question = "타학교 산업공학과 목표로 물리를 안들었는데 건국대 학생부종합 지원하려면 불이익이 클까요? 자유전공학부는요?"
    plan = build_question_plan(question, {"KEYWORD": ["산업공학과"]})
    assert "산업공학과" not in plan.majors + plan.keywords
    assert "자유전공학부" in plan.majors
    assert "판단할 수 없습니다" in plan.advice


@pytest.mark.parametrize("question,intent", [
    ("나 심심해", "smalltalk"), ("연세대랑 고려대중 어디가 더 좋아?", "counseling"),
    ("건대 경영 정시 백분위 몇 정도?", "external_data"),
    ("건국대 경영관 시설 좋은가요?", "campus"),
    ("현대미술과 학생부종합 기타과목 중요도가 궁금해", "clarify"),
])
def test_nondocument_answers_do_not_call_llm(monkeypatch, question, intent):
    def fail(**kwargs):
        pytest.fail("Ungrounded generation must not be called")
    monkeypatch.setattr(answers, "call_llm", fail)
    result = answers.answer_one(question, None, None, None, "", "")
    assert result["stats"]["handler"] == intent
    assert result["answer"]
    assert result["sources"] == []


@pytest.mark.parametrize("question,key,value", [
    ("실기 없는 미대 전형 찾아줘", "no_practical", True),
    ("수능 전형 60%이상인 미대 찾아줘", "minimum_percent", 60),
])
def test_discovery_keeps_constraints(question, key, value):
    plan = build_question_plan(question)
    assert plan.intent == "discovery"
    assert plan.constraints[key] == value


def test_pairs_keep_university_round_relationship():
    pairs = build_pairs_smart("연세대 정시랑 고려대 수시 각각 전형일정", ["연세대", "고려대"], ["정시", "수시"], ["전형일정"])
    assert {(p["UNI"], p["TYPE"]) for p in pairs} == {("연세대", "정시"), ("고려대", "수시")}


def test_ner_is_called_once_per_turn(monkeypatch):
    counts = [0, 0, 0]
    def extractor(index, result):
        def run(text):
            counts[index] += 1
            return result
        return run
    uni = SimpleNamespace(extract_uni=extractor(0, ["건국대"]))
    typ = SimpleNamespace(extract_type=extractor(1, []))
    kw = SimpleNamespace(extract_keywords=extractor(2, ["컴공"]))
    plans = []
    def search(*args, **kwargs):
        plans.append(kwargs["query_plan"])
        return [], {}, {}
    monkeypatch.setattr(answers, "search_top_pages_for_query", search)
    answers.run_single_turn("건대 컴공 논술 최저 있어?", uni, typ, kw, "", "")
    assert counts == [1, 1, 1]
    assert plans[0].rounds == ["수시"]


def test_unknown_round_searches_both_available_rounds(monkeypatch, tmp_path):
    paths = []
    def resolve(uni, round_name):
        paths.append(round_name)
        path = tmp_path / (round_name + ".txt")
        path.write_text("==== Page 1 ====\n지원자격\n고등학교 졸업예정자", encoding="utf-8")
        return str(path)
    monkeypatch.setattr("search_and_export.resolve_text_path", resolve)
    plan = build_question_plan("건국대 지원자격 알려줘")
    search_top_pages_for_query(plan.question, None, None, None, "", "", query_plan=plan)
    assert set(paths) == {"susi", "jungsi"}


def test_dates_normalize_without_splitting_decimal_grades():
    assert _numbers("2025.12.29.") == _numbers("2025년 12월 29일")
    assert _numbers("09.08.") == _numbers("9월 8일")
    assert _numbers("내신 3.0") != _numbers("내신 3")


def test_source_id_accepts_original_table_without_model_requotation():
    pages = {13: "모집단위\n일반전형\n컴퓨터공학부\n36\n경영학과\n53"}
    blocks = evidence_blocks(pages)
    raw = json.dumps({"claims": [{"text": "컴퓨터공학부 일반전형은 36명입니다.", "evidence_ids": ["p13-b0"]}]})
    text, cited = validate_document_answer(raw, pages, blocks=blocks)
    assert "36명" in text
    assert cited == [13]


@pytest.mark.parametrize("text", ["경영학과는 36명입니다.", "컴퓨터공학부는 99명입니다."])
def test_wrong_subject_or_amount_is_rejected(text):
    quote = "컴퓨터공학부 모집인원은 36명입니다."
    raw = json.dumps({"claims": [{"text": text, "evidence": [{"page": 1, "quote": quote}]}]})
    assert validate_document_answer(raw, {1: quote}) == (NO_EVIDENCE, [])


def test_another_rows_number_is_rejected():
    pages = {1: "컴퓨터공학부\n36\n경영학과\n53"}
    raw = json.dumps({"claims": [{"text": "컴퓨터공학부는 53명입니다.", "evidence_ids": ["p1-b0"]}]})
    assert validate_document_answer(raw, pages, blocks=evidence_blocks(pages)) == (NO_EVIDENCE, [])


def test_unknown_evidence_id_is_rejected():
    pages = {1: "컴퓨터공학부 모집인원은 36명입니다."}
    raw = json.dumps({"claims": [{"text": "컴퓨터공학부는 36명입니다.", "evidence_ids": ["p99-b0"]}]})
    assert validate_document_answer(raw, pages, blocks=evidence_blocks(pages)) == (NO_EVIDENCE, [])


@pytest.mark.parametrize("round_name,total", [("susi", "49"), ("jungsi", "36")])
def test_real_seoul_quota_tables_preserve_column_relationship(round_name, total):
    page = 10 if round_name == "susi" else 13
    path = Path(__file__).resolve().parents[1] / "university" / "seoul" / f"{round_name}_text.txt"
    text = answers.load_page_text(str(path), page)
    answer, cited = quota_answer({page: text}, ["컴퓨터공학부"])
    assert f"합계 {total}명" in answer
    assert "일반전형 36명" in answer
    assert cited == [page]


def test_ambiguous_table_does_not_infer_total():
    text = "모집단위\n지역균형전형\n일반전형\n합계\n컴퓨터공학부\n9\n36\n99"
    assert quota_answer({1: text}, ["컴퓨터공학부"]) == ("", [])


def test_another_unit_on_same_line_is_not_evidence():
    pages = {1: "컴퓨터공학부 36명 경영학과 53명"}
    raw = json.dumps({"claims": [{"text": "컴퓨터공학부는 53명입니다.", "evidence_ids": ["p1-b0"]}]})
    assert validate_document_answer(raw, pages, blocks=evidence_blocks(pages)) == (NO_EVIDENCE, [])


@pytest.mark.parametrize("claim", [
    "매체연기학과는 실기 70%로 선발합니다.",
    "미술대학을 제외한 전 모집단위는 수능 100%로 선발합니다.",
])
def test_discovery_rejects_non_art_or_excluded_art(claim):
    pages = {1: claim}
    raw = json.dumps({"claims": [{"text": claim, "evidence_ids": ["p1-b0"]}]})
    assert validate_document_answer(raw, pages, blocks=evidence_blocks(pages), constraints={"art": True}) == (NO_EVIDENCE, [])


def test_flattened_art_table_does_not_prove_document_score_percentage():
    pages = {1: "미술대학 디자인과\n서류평가\n실기평가\n면접\n-\n-\n100"}
    raw = json.dumps({"claims": [{"text": "미술대학 디자인과는 서류평가 100%입니다.", "evidence_ids": ["p1-b0"]}]})
    assert validate_document_answer(raw, pages, blocks=evidence_blocks(pages), constraints={"art": True}) == (NO_EVIDENCE, [])


def test_missing_threshold_does_not_mean_exemption():
    pages = {1: "논술 전형 수능최저학력기준 적용"}
    raw = json.dumps({"claims": [{"text": "논술 전형 수능최저학력기준이 없습니다.", "evidence_ids": ["p1-b0"]}]})
    assert validate_document_answer(raw, pages, blocks=evidence_blocks(pages)) == (NO_EVIDENCE, [])


def test_unrequested_quota_claim_is_not_added_to_method_answer():
    pages = {1: "KU일반학생 인문 자연 420명 수능 100%"}
    raw = json.dumps({"claims": [{"text": "KU일반학생은 수능 100%로 420명을 선발합니다.", "evidence_ids": ["p1-b0"]}]})
    assert validate_document_answer(raw, pages, blocks=evidence_blocks(pages), requested_topics=["전형방법"]) == (NO_EVIDENCE, [])


def test_major_method_cannot_be_replaced_with_unspecified_track():
    claim = "전형요소는 서류 70%, 학생부 30%입니다."
    pages = {1: claim}
    raw = json.dumps({"claims": [{"text": claim, "evidence_ids": ["p1-b0"]}]})
    assert validate_document_answer(raw, pages, blocks=evidence_blocks(pages),
                                    requested_topics=["전형방법"], requested_majors=["자유전공공학부"]) == (NO_EVIDENCE, [])
