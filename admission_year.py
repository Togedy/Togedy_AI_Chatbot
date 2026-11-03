"""Fail closed on unregistered, changed or mismatched admission documents."""
import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

from settings import PROJECT_ROOT
from utils.mapping_loader import uni_to_slug, type_to_slug


CATALOG_PATH = PROJECT_ROOT / "config" / "admission_documents.json"


def catalog():
    try:
        return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))["documents"]
    except (OSError, ValueError, KeyError):
        return []


@lru_cache(maxsize=64)
def _digest(path, size, modified_ns, normalize_newlines=False):
    data = Path(path).read_bytes()
    if normalize_newlines:
        data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def verified_year(path, records=None):
    path = Path(path).resolve()
    records = catalog() if records is None else records
    for record in records:
        if (PROJECT_ROOT / record["text_path"]).resolve() != path:
            continue
        for key in ("text", "pdf"):
            source = PROJECT_ROOT / record[f"{key}_path"]
            if not source.is_file():
                return None
            stat = source.stat()
            if _digest(str(source), stat.st_size, stat.st_mtime_ns, key == "text") != record[f"{key}_sha256"]:
                return None
        return record["academic_year"]
    return None


def year_clarification(plan):
    """Check before reviewed shortcuts as well as ordinary retrieval."""
    if not (plan.topics or plan.intent == "discovery") or plan.intent in ("external_data", "campus"):
        return None
    years = set(re.findall(r"(20\d{2})\s*학년도", plan.question))
    if re.search(r"올해|내년|금년|최신|최근|작년|지난해|전년도|20\d{2}\s*년", plan.question) and not years:
        return "자료의 학년도를 정확히 맞추기 위해 확인할게요. 몇 학년도 모집요강이 궁금하신가요? 예: 2027학년도"
    if len(years) > 1:
        return "학년도별로 자료를 구분해 확인해야 해요. 한 번에 하나의 학년도를 지정해 주세요."
    records = catalog()
    if not records:
        return "모집요강의 학년도 등록 정보를 확인할 수 없어요. 자료 등록 정보를 점검해야 해요."
    unis = {uni_to_slug(u) for u in plan.universities}
    rounds = {type_to_slug(t) for t in plan.rounds}
    selected = [r for r in records if (not unis or Path(r["text_path"]).parent.name in unis)
                and (not rounds or Path(r["text_path"]).stem.removesuffix("_text") in rounds)]
    actual = [verified_year(PROJECT_ROOT / r["text_path"], records) for r in selected]
    if None in actual:
        return "보유 모집요강이 등록된 자료와 달라 학년도를 검증할 수 없어요. 원문과 학년도 정보를 다시 검토해야 해요."
    if not years:
        return None
    requested = int(next(iter(years)))
    if not selected or any(year != requested for year in actual):
        available = sorted({r["academic_year"] for r in selected})
        label = ", ".join(f"{y}학년도" for y in available) or "등록된 학년도 없음"
        return f"요청하신 {requested}학년도와 일치하는 검증된 모집요강을 확보하지 못했어요. 보유 자료: {label}. 확인할 학년도를 다시 알려주세요."
    return None


def source_year_notice(pairs):
    records = catalog()
    paths = {row["doc_path"] for rows in pairs.values() for row in rows if row.get("doc_path")}
    years = {verified_year(p, records) for p in paths}
    if paths and None not in years:
        return "자료 기준: " + ", ".join(f"{y}학년도 모집요강" for y in sorted(years))
    return ""
