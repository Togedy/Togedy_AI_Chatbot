"""Read unambiguous flat quota tables; leave complex/merged tables to abstention.

Only tables with explicit known columns followed by a total are accepted.
Column order, row width and arithmetic must agree; no values are inferred.
"""
import re


def english_grade_answer(pages):
    """Read exactly nine explicit grade columns and their nine point values."""
    found = []
    for page, text in pages.items():
        section = re.search(r"영어\s*등급별\s*환산표(.*?)(?:한국사|====|$)", text, re.S)
        if not section:
            continue
        tokens = [t.strip() for t in section[1].splitlines() if t.strip()]
        try:
            index = tokens.index("등급")
        except ValueError:
            continue
        grades = tokens[index + 1:index + 10]
        if grades != [str(n) for n in range(1, 10)]:
            continue
        tail = tokens[index + 10:]
        if not tail or tail[0] != "공통" or len(tail) < 10:
            continue
        scores = tail[1:10]
        if not all(re.fullmatch(r"\d+(?:\.\d+)?", n) for n in scores):
            continue
        found.append((page, scores))
    if not found or len({tuple(scores) for _, scores in found}) > 1:
        return "", []
    return "영어 등급별 환산점수(공통): " + ", ".join(
        f"{grade}등급 {score}점" for grade, score in enumerate(found[0][1], 1)) + ".", [p for p, _ in found]


def simple_quota_rows(text, majors):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    result = {}
    for major in majors:
        indices = [i for i, line in enumerate(lines) if re.sub(r"\s+", "", line).rstrip("*†") == major]
        for index in indices:
            header = "\n".join(lines[:index])
            total = header.find("합계")
            if total < 0 or "모집단위" not in header:
                continue
            header = header[:total + 2]
            matches = re.findall(r"지역균형\s*전형|일반전형|기회균형\s*특별전형|합계", header)
            columns = [re.sub(r"\s+", "", c) for c in matches]
            if len(columns) < 2 or len(columns) != len(set(columns)) or columns[-1] != "합계":
                continue
            cells = []
            for line in lines[index + 1:]:
                if not re.fullmatch(r"\d+|[·ㆍ-]", line):
                    break
                cells.append(line)
            if len(cells) < len(columns):
                continue
            cells = cells[:len(columns)]
            if not cells[-1].isdigit():
                continue
            if sum(int(v) if v.isdigit() else 0 for v in cells[:-1]) != int(cells[-1]):
                continue
            scope = "정원 내 " if "정원내" in re.sub(r"\s+", "", header) else ""
            result[major] = [f"{scope}{column} {value}명" for column, value in zip(columns, cells) if value.isdigit()]
    return result


def quota_answer(pages, majors):
    found, cited = {}, []
    for page, text in pages.items():
        for major, facts in simple_quota_rows(text, majors).items():
            if major in found and found[major] != facts:
                return "", []  # Conflicting sources are not silently selected.
            found[major] = facts
            if page not in cited:
                cited.append(page)
    if not majors or set(found) != set(majors):
        return "", []
    return "\n".join(f"{major}: {', '.join(found[major])}." for major in majors), cited


def calendar_table_answer(doc_path, pages, tracks=()):
    """Extract complete labelled date rows, without filling merged empty cells.

    Every displayed cell must also occur on its labelled original text page.
    Multiple date columns and rows without an explicit event label are skipped.
    """
    import csv
    from pathlib import Path
    from llm.evidence import _compact

    if tracks:
        return "", []
    path = Path(doc_path)
    if not path.name.endswith("_text.txt"):
        return "", []
    table_path = path.with_name(path.name.replace("_text.txt", "_tables.csv"))
    if not table_path.is_file():
        return "", []
    try:
        with table_path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or "Page" not in reader.fieldnames:
                return "", []
            columns = [c for c in reader.fieldnames if c != "Page"]
            table_rows = list(reader)
    except (OSError, csv.Error):
        return "", []
    date_pattern = re.compile(r"(?:20\d{2}\.\s*)?\d{1,2}\.\s*\d{1,2}\.|\d{1,2}월\s*\d{1,2}일")
    event_pattern = re.compile(r"원서|서류|합격|등록|납부|면접|고사|예비소집|입력|전형료")
    ordered_pages = sorted(pages.items(), key=lambda item: _compact(item[1]).count("전체전형"), reverse=True)
    for page, source in ordered_pages:
        selected, seen, has_header = [], set(), False
        for row in table_rows:
            if row.get("Page") != str(page):
                continue
            cells = [row.get(c) or "" for c in columns]
            compact_cells = [_compact(c) for c in cells]
            if any(c in ("구분", "일시", "일정") for c in compact_cells):
                has_header = True
                continue
            if not has_header:
                continue
            dated = [i for i, c in enumerate(cells) if date_pattern.search(c)]
            if len(dated) != 1:
                continue
            position = dated[0]
            labels = [c for c in cells[:position] if c.strip()]
            if not labels or not event_pattern.search("".join(labels)):
                continue
            # Reject damaged/stale extraction; do not reconstruct missing cells.
            if not all(_compact(c) in _compact(source) for c in labels + [cells[position]]):
                continue
            if _compact("".join(labels) + cells[position]) not in _compact(source):
                continue
            label = " · ".join(" ".join(c.split()) for c in labels)
            value = " ".join(cells[position].split())
            item = (label, value)
            if item not in seen:
                seen.add(item)
                selected.append(item)
        if len(selected) >= 2:
            context = " / ".join(source.strip().splitlines()[:6])
            return ("모집요강 원문 표의 일정 발췌 " + f"(PDF {page}쪽)\n" + context + "\n" +
                    "\n".join(f"- {label}: {value}" for label, value in selected) +
                    "\n서류·시험 일정은 지원 전형과 해당 여부에 따라 적용됩니다. 이 발췌에 없는 세부 일정은 원문을 함께 확인해 주세요."), [page]
    return "", []
