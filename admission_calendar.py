"""Deterministic answers for date-sensitive admission questions."""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Optional
from zoneinfo import ZoneInfo


KST = ZoneInfo("Asia/Seoul")

# Announced by the Ministry of Education. Keys are the calendar year in which
# the exam is held, not the academic year printed on the exam name.
CSAT_DATES = {
    2026: date(2026, 11, 19),  # 2027학년도 수능
    2027: date(2027, 11, 18),  # 2028학년도 수능
}

MOE_CSAT_SOURCE = "https://www.moe.go.kr/boardCnts/viewRenew.do?boardID=294&boardSeq=100526&lev=0&m=020402&opType=N&s=moe&statusYN=W"


def today_kst() -> date:
    return datetime.now(KST).date()


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").lower())


def is_current_date_question(text: str) -> bool:
    compact = _compact(text)
    patterns = (
        "오늘날짜", "오늘며칠", "오늘몇일", "오늘이몇월며칠", "오늘몇월며칠",
        "현재날짜", "현재며칠", "지금날짜", "지금며칠", "지금몇월며칠",
    )
    return any(pattern in compact for pattern in patterns)


def is_csat_date_question(text: str) -> bool:
    compact = _compact(text)
    if not any(term in compact for term in ("수능", "대학수학능력시험")):
        return False
    return any(term in compact for term in ("언제", "날짜", "며칠", "몇일", "얼마", "남았", "디데이", "d-day", "d데이"))


def _requested_exam_year(text: str, today: date) -> int:
    match = re.search(r"(20\d{2})\s*(학)?년", text or "")
    if match:
        year = int(match.group(1))
        # “2027학년도 수능” is held in 2026, while “2027년 수능” is held in 2027.
        if match.group(2) or "학년도" in (text or ""):
            return year - 1
        return year

    if today.year in CSAT_DATES and today <= CSAT_DATES[today.year]:
        return today.year
    future_years = sorted(year for year, exam_date in CSAT_DATES.items() if exam_date > today)
    return future_years[0] if future_years else today.year


def answer_calendar_question(text: str, *, today: Optional[date] = None) -> Optional[str]:
    """Return a factual answer for supported date questions, otherwise None."""
    current = today or today_kst()

    if is_current_date_question(text):
        weekdays = "월화수목금토일"
        return f"오늘은 {current.year}년 {current.month}월 {current.day}일 {weekdays[current.weekday()]}요일입니다."

    if not is_csat_date_question(text):
        return None

    exam_year = _requested_exam_year(text, current)
    exam_date = CSAT_DATES.get(exam_year)
    if exam_date is None:
        return "요청하신 연도의 수능 시행일은 현재 등록된 공식 일정에서 확인할 수 없습니다. 교육부 또는 한국교육과정평가원 공고를 확인해 주세요."

    academic_year = exam_year + 1
    days = (exam_date - current).days
    date_text = f"{exam_date.year}년 {exam_date.month}월 {exam_date.day}일"
    if days > 0:
        timing = f"오늘({current.year}년 {current.month}월 {current.day}일) 기준으로 {days}일 남았습니다."
    elif days == 0:
        timing = "오늘이 수능 시험일입니다."
    else:
        timing = f"해당 시험은 {-days}일 전에 시행됐습니다."

    return f"{academic_year}학년도 수능은 {date_text}에 시행됩니다. {timing} 교육부 발표 기준입니다."
