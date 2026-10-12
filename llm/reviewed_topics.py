"""Topic lookup over reviewed, source-locked tables, with explicit scope.

These records are maintained admissions data, not query/answer fixtures. A
record is unusable until every referenced source page has a matching digest.
"""
import json
from pathlib import Path

from llm.reviewed_facts import ROOT, source_digest

INDEX = ROOT / "config/reviewed_topic_tables.json"


def load_topics(index=INDEX, root=ROOT):
    from generate_answers import load_page_text
    try:
        records = json.loads(Path(index).read_text(encoding="utf-8"))["records"]
    except (OSError, ValueError, KeyError):
        return []
    result = []
    for record in records:
        valid = bool(record.get("sources")) and bool(record.get("rows"))
        for source in record.get("sources", []):
            path = (root / source["document"]).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file():
                valid = False
                break
            text = load_page_text(str(path), source["page"])
            if not text or source_digest(text) != source.get("sha256"):
                valid = False
                break
        if valid:
            result.append(record)
    return result


def select_topic(records, plan, university, round_name, topic):
    matches = []
    for r in records:
        if (r["university"], r["round"], r["topic"]) != (university, round_name, topic):
            continue
        if any(term in plan.question.replace(" ", "") for term in r.get("excluded_terms", [])):
            continue
        if plan.tracks and (not r.get("tracks") or not all(t in r["tracks"] for t in plan.tracks)):
            continue
        if plan.majors and topic in ("전형방법", "서류평가", "수능최저"):
            if not all(m in r.get("majors", []) for m in plan.majors):
                continue
        matches.append(r)
    if len(matches) != 1:
        return None
    return matches[0]


def render_topic(record):
    lines = [record["scope"]]
    lines.extend(f"- {label}: {value}" for label, value in record["rows"])
    if record.get("notes"):
        lines.append(record["notes"])
    return "\n".join(lines)


def source_rows(record):
    return [{"matched_uni": record["university"], "matched_type": record["round"],
             "doc_path": str(ROOT / s["document"]), "page_index": s["page"],
             "score": 1.0, "snippet": record["scope"]} for s in record["sources"]]
