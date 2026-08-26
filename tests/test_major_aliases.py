from utils.major_aliases import expand_major_keywords, discover_document_major_aliases
from generate_answers import build_missing_major_answer, extract_requested_major_terms


def test_yonsei_computer_science_expands_to_admission_unit():
    keywords, notes = expand_major_keywords(
        "yonsei",
        ["컴퓨터과학과", "모집인원"],
        question="연세대 컴퓨터과학과 모집인원",
    )

    assert "컴퓨터과학과" in keywords
    assert "첨단컴퓨팅학부" in keywords
    assert any("2학년 진급" in note for note in notes)


def test_alias_is_university_specific():
    keywords, notes = expand_major_keywords(
        "korea",
        ["컴퓨터과학과"],
        question="고려대 컴퓨터과학과",
    )

    assert keywords == ["컴퓨터과학과"]
    assert notes == []


def test_discovers_konkuk_computer_engineering_unit_from_document():
    keywords, notes = discover_document_major_aliases(
        ["컴퓨터과학과", "모집인원"],
        question="건국대학교 수시 컴퓨터과학과 모집인원",
        document_text="공과대학\n컴퓨터공학부\n자연\n100",
    )

    assert "컴퓨터공학부" in keywords
    assert any("컴퓨터공학부" in note for note in notes)


def test_does_not_replace_an_exact_document_unit():
    keywords, notes = discover_document_major_aliases(
        ["컴퓨터과학과"],
        question="컴퓨터과학과 모집인원",
        document_text="컴퓨터과학과\n컴퓨터공학부",
    )

    assert keywords == ["컴퓨터과학과"]
    assert notes == []


def test_extracts_major_from_question_when_ner_misses_it():
    terms = extract_requested_major_terms(
        "건국대학교 수시 양자컴퓨터학과 모집인원을 알려줘",
        ["모집인원"],
    )
    assert terms == ["양자컴퓨터학과"]


def test_generic_missing_major_message():
    answer = build_missing_major_answer("건국대", ["양자컴퓨터학과"])
    assert "'양자컴퓨터학과'" in answer
    assert "건국대 모집요강에서 찾을 수 없습니다" in answer
    assert "다른 학과·학부 또는 모집단위 명칭" in answer
