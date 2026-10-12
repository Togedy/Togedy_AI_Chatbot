"""Validate document-answer citations before exposing generated claims."""

import json
import re
from query_constraints import text_satisfies_percentages


NO_EVIDENCE = "제공된 모집요강 발췌에서는 해당 내용을 확인하지 못했습니다."
EVIDENCE_INSTRUCTIONS = """
[응답 형식]
아래 JSON 형식이 최종 출력 형식입니다. 앞선 자연어 작성 원칙은 각 text 필드에 적용하세요.
JSON 객체만 반환하세요: {"claims": [{"text": "확인된 사실 한 문장", "evidence_ids": ["p13-b0"]}], "unanswered": []}
evidence_ids는 제공된 근거 구간의 ID만 사용하세요. 인용문을 새로 작성하지 마세요.
각 문장은 해당 질문의 대학·전형·모집단위에 직접 대응해야 합니다.
세부 전형이 지정되지 않았으면 확인한 전형명을 각 문장에 명시하고 대학 전체의 공통 조건으로 일반화하지 마세요.
각 근거는 주장을 뒷받침하는 항목명, 조건, 표의 행·열 제목과 값을 함께 포함해야 합니다.
표의 행·열 연결을 확인할 수 없으면 해당 주장을 생략하세요.
검색용 별칭이나 명칭 참고 설명 자체는 문서 근거가 아닙니다.
직접 모집단위가 다르면 같은 학과로 단정하지 말고 실제 문서의 명칭을 사용하세요.
숫자는 근거의 값을 사용하고, 합산하거나 새 숫자를 계산하지 마세요. 근거에 없는 연도는 붙이지 마세요.
모집인원은 실제 모집단위명과 열의 전형명 또는 합계 여부를 반드시 문장에 포함하세요.
질문한 모든 항목을 점검하고 답하지 못한 항목명은 unanswered에 기재하세요.
확인할 수 없는 경우 {"claims": []}를 반환하세요. JSON 밖의 설명은 쓰지 마세요.
"""


def _compact(text):
    return re.sub(r"[\s\x00-\x1f]+", "", text)


def _numbers(text):
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    # A trailing dot identifies a date, unlike decimal grades such as 3.0.
    text = re.sub(r"(?<!\d)(?:\d{4}\.\s*)?\d{1,2}\.\s*\d{1,2}\.(?!\d)",
                  lambda m: " ".join(re.findall(r"\d+", m[0])), text)
    return {str(int(n)) if "." not in n else n for n in re.findall(r"\d+(?:\.\d+)?", text)}


def evidence_blocks(pages):
    """Keep a complete PDF page: a table header and its rows cannot be split.

    Pages are already selected by retrieval. A single ID also avoids duplicated
    overlapping text and accidental citations of the wrong window on a page.
    """
    return {f"p{page}-b0": {"page": page, "text": text} for page, text in pages.items()}


def _subjects(text):
    return set(re.findall(r"[가-힣A-Za-z·]+(?:학과|학부|전공|예과|화과|술과|조소과|디자인과|교육과)", text))


def _relations_match(text, quotes, university=""):
    source = "\n".join(quotes)
    normalized = _compact(source)
    components = re.findall(r"[가-힣]+역량\s*\(?\s*(\d[\d,]*)\s*점", text)
    total = re.search(r"총\s*([\d,]+)\s*점", text)
    if total and len(components) > 1:
        if sum(int(n.replace(",", "")) for n in components) != int(total[1].replace(",", "")):
            return False
    if re.search(r"수능최저(?:학력기준)?[^.!?]{0,12}(?:없|미적용|적용하지않)", _compact(text)):
        # Absence of a condition in the excerpt is not evidence of exemption.
        if not re.search(r"수능최저(?:학력기준)?[^.!?]{0,12}(?:없|미적용|적용하지않)", normalized):
            return False
    # Matching a number somewhere on a page cannot prove a different unit's claim.
    for subject in _subjects(text):
        if _compact(subject) not in normalized:
            return False
        amounts = re.findall(r"(\d[\d,]*)\s*명", text)
        if amounts:
            lines = [line.strip() for line in source.splitlines() if line.strip()]
            local_numbers = set()
            for i, line in enumerate(lines):
                if _compact(subject) not in _compact(line):
                    continue
                start = _compact(line).find(_compact(subject))
                segment = _compact(line)[start:]
                others = _subjects(segment) - {subject}
                end = min((segment.find(s) for s in others if segment.find(s) > 0), default=len(segment))
                row = [segment[:end]]
                if end < len(segment):
                    local_numbers |= _numbers(row[0])
                    continue
                for next_line in lines[i + 1:i + 16]:
                    if _subjects(next_line) - {subject}:
                        break
                    row.append(next_line)
                local_numbers |= _numbers("\n".join(row))
            if not {n.replace(",", "") for n in amounts} <= local_numbers:
                return False
    if university:
        from question_plan import UNIVERSITIES
        for canonical, aliases in UNIVERSITIES.items():
            if canonical != university and any(alias in text for alias in aliases):
                return False
    return True


def validate_document_answer(raw, pages, *, blocks=None, university="", constraints=None, requested_topics=None,
                             requested_majors=None):
    """Return only claims with real quotations and numbers present in them.

    This checks provenance, not semantic entailment; the prompt still requires
    the model to check the meaning and table structure of every quotation.
    """
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```json") and cleaned.endswith("```"):
            cleaned = cleaned[7:-3].strip()
        payload = json.loads(cleaned)
    except (ValueError, TypeError, AttributeError):
        return NO_EVIDENCE, []
    if not isinstance(payload, dict) or not isinstance(payload.get("claims"), list):
        return NO_EVIDENCE, []

    answers, used_pages = [], []
    rejected = False
    for claim in payload["claims"]:
        if not isinstance(claim, dict):
            rejected = True
            continue
        text, evidence = claim.get("text"), claim.get("evidence")
        if blocks is not None and "evidence_ids" in claim:
            ids = claim["evidence_ids"]
            if not isinstance(ids, list) or not ids or any(not isinstance(i, str) or i not in blocks for i in ids):
                rejected = True
                continue
            evidence = [{"page": blocks[i]["page"], "quote": blocks[i]["text"]} for i in ids]
        if not isinstance(text, str) or not text.strip() or not isinstance(evidence, list) or not evidence:
            rejected = True
            continue
        if requested_topics is not None and "모집인원" not in requested_topics and re.search(r"\d[\d,]*\s*명", text):
            # Unrequested quota details can mistake a table subtotal for the total.
            rejected = True
            continue
        if requested_majors and requested_topics and "전형방법" in requested_topics:
            if not any(_compact(major) in _compact(text) for major in requested_majors):
                rejected = True
                continue
        quotes, cited = [], []
        for item in evidence:
            if not isinstance(item, dict):
                break
            page, quote = item.get("page"), item.get("quote")
            if type(page) is not int or page not in pages or not isinstance(quote, str):
                break
            from_block = blocks is not None and "evidence_ids" in claim
            if len(_compact(quote)) < 8 or (not from_block and _compact(quote) not in _compact(pages[page])):
                break
            quotes.append(quote)
            cited.append(page)
        else:
            constraint_ok = True
            if constraints and constraints.get("art"):
                # A page mentioning art does not make every unit on it an art unit.
                constraint_ok = bool(re.search(r"미술대학|회화|미술과|디자인|조소|조형|공예", text))
                constraint_ok = constraint_ok and not bool(re.search(
                    r"(?:미술대학|미술|회화|디자인|조소|조형|공예).{0,15}(?:제외|이외)", text))
            if constraints:
                constraint_ok = constraint_ok and text_satisfies_percentages(text, constraints)
                if constraints.get("stage"):
                    constraint_ok = constraint_ok and constraints["stage"] in _compact(text)
                constraint_ok = constraint_ok and university not in constraints.get("excluded_universities", [])
                constraint_ok = constraint_ok and not any(
                    _compact(m) in _compact(text) for m in constraints.get("excluded_majors", []))
            if constraints and constraints.get("no_practical"):
                constraint_ok = constraint_ok and bool(re.search(r"실기.{0,10}(?:없|미실시|하지 않)|(?:수능|서류|학생부).{0,8}100\s*%", text))
            if constraints and constraints.get("art"):
                # Flattened multi-stage tables cannot prove a component-to-column
                # relationship merely by containing the same percentage.
                ratios = re.findall(r"(수능|실기|서류|학생부)(?:평가)?[^\d\n%]{0,12}(\d+(?:\.\d+)?)\s*%", text)
                source = "\n".join(quotes)
                for component, ratio in ratios:
                    if not re.search(re.escape(component) + r"(?:평가)?[^\d\n%]{0,12}" + re.escape(ratio) + r"\s*%", source):
                        constraint_ok = False
            if constraint_ok and _numbers(text) <= _numbers("\n".join(quotes)) and _relations_match(text, quotes, university):
                if text.strip() not in answers:
                    answers.append(text.strip())
                used_pages.extend(page for page in cited if page not in used_pages)
                continue
        rejected = True

    if not answers:
        return NO_EVIDENCE, []
    if rejected or payload.get("unanswered"):
        answers.append("질문하신 내용 중 일부는 발췌 근거로 확인하지 못했습니다.")
    return "\n".join(answers), used_pages
