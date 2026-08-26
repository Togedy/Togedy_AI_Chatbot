from datetime import date

from admission_calendar import answer_calendar_question


TODAY = date(2026, 8, 21)


def test_answers_current_date_without_llm():
    assert answer_calendar_question("지금 날짜가 어떻게 돼?", today=TODAY) == (
        "오늘은 2026년 8월 21일 금요일입니다."
    )


def test_answers_csat_countdown_without_llm():
    answer = answer_calendar_question("수능까지 얼마 남았어?", today=TODAY)
    assert answer is not None
    assert "2027학년도 수능" in answer
    assert "2026년 11월 19일" in answer
    assert "90일 남았습니다" in answer


def test_understands_academic_year():
    answer = answer_calendar_question("2027학년도 수능 날짜 알려줘", today=TODAY)
    assert answer is not None
    assert "2026년 11월 19일" in answer


def test_ignores_non_date_csat_question():
    assert answer_calendar_question("수능이 무엇인가요?", today=TODAY) is None
