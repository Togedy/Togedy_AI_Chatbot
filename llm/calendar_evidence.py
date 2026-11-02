"""Preserve explicit one-column schedule rows and start/end time relations."""
import re


DATE_LINE = re.compile(r"^(?:20\d{2}\.\s*)?\d{1,2}\.\s*\d{1,2}\.[0-9가-힣\s().:~～\-]*$")
EVENT = re.compile(
    r"^(?:지원서접수(?:\(인터넷접수\))?|서류제출(?:\(해당자\))?|학교장추천확인|"
    r"1단계합격자발표|2단계전형료납부|합격자발표|합격자등록\(온라인문서등록\)|"
    r"\d차충원합격자(?:발표|등록\(온라인문서등록\))|합격자최종등록\(등록금납부\))$"
)


def schedule_rows(source):
    lines = [line.strip() for line in source.splitlines() if line.strip()]
    rows = []
    for label, value in zip(lines, lines[1:]):
        label = label.rstrip("*†")
        if EVENT.fullmatch(re.sub(r"\s+", "", label)) and DATE_LINE.fullmatch(value):
            rows.append((label, value))
    return rows


def calendar_text_answer(pages, tracks=()):
    for page, source in pages.items():
        if "입학전형 일정표" not in source:
            continue
        scopes = list(dict.fromkeys(re.findall(r"학생부종합전형\([^\n)]+\)", source)))
        if len(scopes) != 1 or (tracks and not all(t in scopes[0] for t in tracks)):
            continue
        rows = schedule_rows(source)
        if len(rows) < 3:
            continue
        return (
            f"모집요강 원문 일정 발췌 (PDF {page}쪽)\n적용 전형: {scopes[0]}\n"
            + "\n".join(f"- {label}: {value}" for label, value in rows)
            + "\n해당 전형의 원문에서 확인한 일정이며 다른 전형에는 그대로 적용하지 않아요.",
            [page],
        )
    return "", []


def calendar_claim_matches(text, sources):
    """Reject a claimed closing time that is a row's opening time instead."""
    labels = list(re.finditer(r"지원서\s*접수|서류\s*제출", text))
    rows = schedule_rows(sources)
    for index, match in enumerate(labels):
        segment = text[match.end():labels[index+1].start() if index+1 < len(labels) else len(text)]
        closings = re.findall(r"(\d{1,2}:\d{2})\s*(?:까지|마감|종료)", segment)
        if not closings:
            continue
        name = re.sub(r"\s+", "", match[0])
        values = [v for label, v in rows if re.sub(r"\s+", "", label).startswith(name)]
        expected = {times[-1] for value in values if (times := re.findall(r"\d{1,2}:\d{2}", value))}
        if len(expected) == 1 and any(t not in expected for t in closings):
            return False
    return True
