# postprocess.py
# UNI 전용 후처리:
# - B-UNI/I-UNI 스팬 병합
# - 조사/문장부호 제거
# - 축약형/동의형 → 표준 대학명 매핑
# - 복합 축약형(연고대/서연고) → 다중 대학 확장
# - 중복 제거(순서 보존)

from typing import List, Dict
import re

# 1) 표준 대학명 목록
CANONICALS = ["고려대", "연세대", "성균관대", "한양대", "건국대", "서울대", "서강대"]

# 2) 축약형/동의형 → 표준명 매핑 (공백 제거 기준)
ALIAS_TO_CANON = {
    # 고려대
    "고려대": "고려대", "고대": "고려대", "고려대학교": "고려대", "고려대학": "고려대",
    # 연세대
    "연세대": "연세대", "연대": "연세대", "연세대학교": "연세대", "연세대학": "연세대",
    # 성균관대
    "성균관대": "성균관대", "성대": "성균관대", "성균관": "성균관대", "성균관대학교": "성균관대", "성균관대학": "성균관대",
    # 한양대
    "한양대": "한양대", "한양": "한양대", "한양대학교": "한양대", "한양대학": "한양대",
    # 건국대
    "건국대": "건국대", "건대": "건국대", "건국대학교": "건국대", "건국대학": "건국대",
    # 서울대
    "서울대": "서울대", "설대": "서울대", "서울대학교": "서울대",
    # 서강대
    "서강대": "서강대", "서강": "서강대", "서강대학교": "서강대", "서강대학": "서강대",
}

# 3) 복합 축약형 → 다중 표준명 확장
COMPOSITE_MAP = {
    "서연고": ["서울대", "연세대", "고려대"],
    "고연대": ["연세대", "고려대"],
    "연고대": ["연세대", "고려대"],
    "서성한": ["서강대", "성균관대","한양대"]
}

# 4) 조사/문장부호 제거
JOSA_SUFFIXES = [
    "은", "는", "이", "가", "을", "를", "과", "와", "랑", "도", "에서", "의", "까지", "만",
    "는요", "인가요", "입니다"
]
JOSA_SUFFIXES = sorted(JOSA_SUFFIXES, key=len, reverse=True)
RX_TRAIL = re.compile(r"[^\w가-힣]+$")

def _strip_suffixes(s: str) -> str:
    s = RX_TRAIL.sub("", s)   # 끝 문장부호 제거
    for suf in JOSA_SUFFIXES: # 조사/어미 제거
        if s.endswith(suf):
            s = s[: -len(suf)]
            break
    return s

def _to_canonical(raw: str) -> List[str]:
    if not raw:
        return []
    s = raw.replace(" ", "")
    s = _strip_suffixes(s)

    # '대학교/대학' → '대' 정규화
    if s.endswith("대학교"):
        s = s[:-3] + "대"
    elif s.endswith("대학"):
        s = s[:-2] + "대"

    # 복합 축약형
    if s in COMPOSITE_MAP:
        return COMPOSITE_MAP[s][:]

    # 단일 축약형
    if s in ALIAS_TO_CANON:
        return [ALIAS_TO_CANON[s]]

    return []  # 매핑 실패 시 폐기(정밀도 우선)

def _dedup_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def postprocess_ner_output(tokens: List[str], tags: List[str]) -> Dict[str, List[str]]:
    assert len(tokens) == len(tags), "tokens and tags must align"

    raw_ents: List[str] = []
    cur = []
    for t, y in zip(tokens, tags):
        if y == "B-UNI":
            if cur:
                raw_ents.append("".join(cur))
                cur = []
            cur = [t]
        elif y == "I-UNI":
            if cur:
                cur.append(t)
            else:
                cur = [t]  # 단독 I 방어적 처리
        else:
            if cur:
                raw_ents.append("".join(cur))
                cur = []
    if cur:
        raw_ents.append("".join(cur))

    canon_list: List[str] = []
    for ent in raw_ents:
        canon_list.extend(_to_canonical(ent))

    canon_list = _dedup_preserve_order(canon_list)
    canon_list = [c for c in canon_list if c in CANONICALS]
    return {"UNI": canon_list}
