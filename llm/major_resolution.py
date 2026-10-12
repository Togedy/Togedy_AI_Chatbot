"""Clarify unrecognized unit names without silently treating them as aliases."""
from difflib import SequenceMatcher
from pathlib import Path
import re


def major_name_clarification(plan):
    if len(plan.universities) != 1 or len(plan.majors) != 1 or "전형방법" not in plan.topics:
        return None
    from search_and_export import get_uni_slug, resolve_text_path
    slug = get_uni_slug(plan.universities[0])
    documents = [Path(resolve_text_path(slug, r)) for r in ("susi", "jungsi")]
    texts = [p.read_text(encoding="utf-8") for p in documents if p.is_file()]
    target = plan.majors[0]
    if not texts or any(target in re.sub(r"\s+", "", t) for t in texts):
        return None
    # Standalone table cells only; prose mentioning a department is not a name.
    names = set()
    for text in texts:
        for line in text.splitlines():
            line = line.strip().rstrip("*†")
            if re.fullmatch(r"[가-힣A-Za-z·]{3,24}(?:학부|학과)", line):
                names.add(line)
    def similarity(name):
        score = SequenceMatcher(None, target, name).ratio()
        # Use an engineering hint to rank suggestions, never as an alias proof.
        if "공학" in target and "공과" in name:
            score += 0.2
        return score
    scored = sorted(((similarity(name), name) for name in names), reverse=True)
    candidates = [name for score, name in scored if score >= 0.6][:3]
    if not candidates:
        return None
    return f"보유 모집요강에서 ‘{target}’이라는 정확한 모집단위명은 확인되지 않습니다. " + " / ".join(candidates) + " 중 어느 모집단위를 말씀하신 건가요? 모집단위에 따라 지원 전형이 달라집니다."
