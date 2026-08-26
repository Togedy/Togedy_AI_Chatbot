"""University-specific admission-unit aliases used for retrieval expansion."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALIAS_PATH = PROJECT_ROOT / "config" / "major_aliases.json"

# Concept groups bridge common user terminology to the exact unit name that
# appears in each university's own admission guide. A term is only added when
# it is also found in that university/type document, so no school inherits
# another school's organization.
MAJOR_CONCEPT_GROUPS = (
    (
        "컴퓨터과학과", "컴퓨터과학부", "컴퓨터공학과", "컴퓨터공학부",
        "컴퓨터학과", "컴퓨터소프트웨어학부", "소프트웨어학과",
        "지능형소프트웨어학과",
    ),
)


def _normalize(value: str) -> str:
    return re.sub(r"\s+", "", (value or "")).lower()


@lru_cache(maxsize=1)
def load_major_aliases(path: str = str(DEFAULT_ALIAS_PATH)) -> Dict:
    alias_path = Path(path)
    if not alias_path.is_file():
        return {}
    with alias_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def expand_major_keywords(
    uni_slug: str,
    keywords: Iterable[str],
    *,
    question: str = "",
) -> Tuple[List[str], List[str]]:
    """Return original keywords plus official admission-unit search aliases.

    The relation notes are kept separately so an alias is never presented as
    if it were the exact same department name.
    """
    expanded = [str(value).strip() for value in keywords if str(value).strip()]
    notes: List[str] = []
    searchable = _normalize(" ".join([question, *expanded]))

    for requested_name, alias in load_major_aliases().get(uni_slug, {}).items():
        if _normalize(requested_name) not in searchable:
            continue
        for term in alias.get("search_terms", []):
            clean_term = str(term).strip()
            if clean_term and clean_term not in expanded:
                expanded.append(clean_term)
        note = str(alias.get("relation", "")).strip()
        if note and note not in notes:
            notes.append(note)

    return expanded, notes


def discover_document_major_aliases(
    keywords: Iterable[str],
    *,
    question: str,
    document_text: str,
) -> Tuple[List[str], List[str]]:
    """Discover a school's exact major name from its selected document.

    This complements curated relationship mappings. It never invents a unit:
    candidates must occur in the university admission document itself.
    """
    expanded = [str(value).strip() for value in keywords if str(value).strip()]
    notes: List[str] = []
    searchable = _normalize(" ".join([question, *expanded]))
    normalized_document = _normalize(document_text)

    for group in MAJOR_CONCEPT_GROUPS:
        requested = next((term for term in group if _normalize(term) in searchable), None)
        if not requested:
            continue

        document_units = [
            term for term in group
            if _normalize(term) in normalized_document
        ]
        exact_requested_exists = _normalize(requested) in normalized_document
        alternatives = [term for term in document_units if _normalize(term) != _normalize(requested)]

        # If the exact requested unit exists, normal retrieval is already the
        # safest choice. Curated mappings handle special cases such as Yonsei's
        # integrated admission and later major selection.
        if exact_requested_exists or not alternatives:
            continue

        for term in alternatives:
            if term not in expanded:
                expanded.append(term)

        alternatives_text = ", ".join(alternatives)
        notes.append(
            f"요청한 '{requested}' 명칭은 이 모집요강에서 직접 확인되지 않으며, "
            f"관련 모집단위로 '{alternatives_text}' 명칭이 확인됩니다."
        )

    return expanded, notes
