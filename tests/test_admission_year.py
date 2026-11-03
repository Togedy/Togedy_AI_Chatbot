from types import SimpleNamespace

import admission_year as years


def plan(question):
    return SimpleNamespace(question=question, topics=["모집인원"], intent="fact",
                           universities=["서울대"], rounds=["수시"])


def test_wrong_year_and_relative_year_are_clarified():
    assert years.year_clarification(plan("2027학년도 서울대 수시 모집인원"))
    assert years.year_clarification(plan("최신 서울대 수시 모집인원"))
    assert years.year_clarification(plan("2026학년도 서울대 수시 모집인원")) is None


def test_missing_catalog_and_ambiguous_calendar_year_are_clarified(monkeypatch):
    assert years.year_clarification(plan("2026년 서울대 수시 모집인원"))
    assert years.year_clarification(plan("작년 서울대 수시 모집인원"))
    monkeypatch.setattr(years, "CATALOG_PATH", years.PROJECT_ROOT / "tmp/missing-catalog.json")
    assert years.year_clarification(plan("서울대 수시 모집인원"))


def test_changed_document_is_not_accepted(tmp_path, monkeypatch):
    text = tmp_path / "susi_text.txt"
    pdf = tmp_path / "susi.pdf"
    text.write_text("2026학년도", encoding="utf-8")
    pdf.write_bytes(b"original")
    import hashlib
    record = {"academic_year": 2026, "text_path": str(text), "pdf_path": str(pdf),
              "text_sha256": hashlib.sha256(text.read_bytes()).hexdigest(),
              "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest()}
    assert years.verified_year(text, [record]) == 2026
    pdf.write_bytes(b"updated source")
    assert years.verified_year(text, [record]) is None


def test_yonsei_forecast_does_not_change_document_year():
    rows = {("연세대", "수시"): [{"doc_path": str(years.PROJECT_ROOT / "university/yonsei/susi_text.txt")}]}
    assert years.source_year_notice(rows) == "자료 기준: 2026학년도 모집요강"


def test_year_guard_precedes_reviewed_shortcuts(monkeypatch):
    from llm.document_pipeline import answer_with_plan
    monkeypatch.setattr("llm.document_pipeline.discovery_answer", lambda *a: (_ for _ in ()).throw(AssertionError("shortcut ran")))
    result = answer_with_plan("2027학년도 서울대 수시 모집인원", None, None, None, "", "",
                              ner_override={"UNI": ["서울대"], "TYPE": ["수시"], "KEYWORD": ["모집인원"]})
    assert result["decision"] == "재질문"
    assert not result["pair_to_rows"]


def test_followup_keeps_year_but_accepts_correction():
    from generate_answers import build_followup_resolved_question
    ner = {"UNI": ["서울대"], "TYPE": ["수시"], "KEYWORD": ["모집인원"]}
    assert "2027학년도" in build_followup_resolved_question("2027학년도 서울대 모집인원", "수시", ner, ner)
    corrected = build_followup_resolved_question("2027학년도 서울대 모집인원", "2026학년도로 알려줘", ner, ner)
    assert "2026학년도" in corrected and "2027학년도" not in corrected


def test_text_hash_accepts_cross_platform_newlines(tmp_path):
    import hashlib
    text, pdf = tmp_path / "text.txt", tmp_path / "doc.pdf"
    text.write_bytes(b"2026\r\nadmission\r\n")
    pdf.write_bytes(b"pdf")
    record = {"academic_year": 2026, "text_path": str(text), "pdf_path": str(pdf),
              "text_sha256": hashlib.sha256(b"2026\nadmission\n").hexdigest(),
              "pdf_sha256": hashlib.sha256(b"pdf").hexdigest()}
    assert years.verified_year(text, [record]) == 2026
    text.write_bytes(b"2026\nadmission\n")
    assert years.verified_year(text, [record]) == 2026
