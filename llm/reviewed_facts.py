"""Source-locked structured records for manually reviewed merged tables.

The reviewed data is a small, explicit corpus index, never a list of questions
or model answers. New records require checking all stages in the original PDF.
"""
import hashlib
import json
from pathlib import Path
import re
from query_constraints import conditions, satisfies

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "config" / "reviewed_admission_facts.json"


def source_digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def verified_records(index=INDEX, root=ROOT):
    from generate_answers import load_page_text
    try:
        records = json.loads(Path(index).read_text(encoding="utf-8"))["facts"]
    except (OSError, ValueError, KeyError):
        return []
    valid = []
    for record in records:
        path = (root / record["document"]).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            continue
        source = load_page_text(str(path), record["page"])
        compact = re.sub(r"\s+", "", source)
        if source_digest(source) != record.get("source_sha256"):
            continue
        if not all(re.sub(r"\s+", "", a) in compact for a in record["anchors"]):
            continue
        stages = record.get("stages", [])
        if not stages or any(sum(s["components"].values()) != 100 for s in stages):
            continue
        if not record["practical_required"] and any("실기" in c for s in stages for c in s["components"]):
            continue
        valid.append(record)
    return valid


def matches(record, plan):
    if record["university"] in plan.constraints.get("excluded_universities", []):
        return False
    if any(re.sub(r"\s+", "", m) in re.sub(r"\s+", "", record["major"])
           for m in plan.constraints.get("excluded_majors", [])):
        return False
    if plan.universities and record["university"] not in plan.universities:
        return False
    if plan.rounds and record["round"] not in plan.rounds:
        return False
    if plan.majors and not any(m in record["major"] for m in plan.majors):
        return False
    if plan.tracks and not all(t in record["track"] for t in plan.tracks):
        return False
    if plan.constraints.get("no_practical") and record["practical_required"]:
        return False
    stage = plan.constraints.get("stage")
    if stage:
        selected = [s for s in record["stages"] if s["name"] == stage]
        if len(selected) != 1:
            return False
        final = selected[0]["components"]
    else:
        final = record["stages"][-1]["components"]
    for condition in conditions(plan.constraints):
        values = [v for c, v in final.items() if condition["component"] in c]
        # Do not invent a zero percentage for an unlisted selection component.
        if not values or not satisfies(sum(values), condition):
            return False
    return True


def discovery_answer(plan):
    if plan.intent != "discovery" or any(t != "전형방법" for t in plan.topics):
        return None
    records = [r for r in verified_records() if matches(r, plan)]
    if not records:
        return None
    if plan.universities and not set(plan.universities) <= {r["university"] for r in records}:
        return None
    if plan.majors and not all(any(m in r["major"] for r in records) for m in plan.majors):
        return None
    parts = ["보유 모집요강에서 확인한 사례입니다. 전국 전형 전체 목록은 아닙니다."]
    rows = {}
    for r in records:
        stages = []
        for s in r["stages"]:
            stages.append(f"{s['name']}: " + " + ".join(f"{c} {v}%" for c, v in s["components"].items()) + f" ({s['selection']})")
        practical = "실기 있음" if r["practical_required"] else "실기 없음"
        parts.append(f"{r['university']} {r['round']} · {r['track']} · {r['major']}\n{practical}. " + "; ".join(stages) + ".\n" + r["notes"])
        row = {"matched_uni": r["university"], "matched_type": r["round"],
               "doc_path": str(ROOT / r["document"]), "page_index": r["page"], "score": 1.0,
               "snippet": r["major"] + " " + r["track"]}
        rows.setdefault((r["university"], r["round"]), []).append(row)
    return "\n\n".join(parts), rows
