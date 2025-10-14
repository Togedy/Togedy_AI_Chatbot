# lexicon.py
from typing import List, Tuple
import re

# 7개 표준 대학명
CANONICALS = ["고려대","연세대","성균관대","한양대","건국대","서울대","서강대"]

# 축약형/동의형 → 표준명 (공백 제거 기준)
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
# 복합 축약형 → 다중 학교
COMPOSITES = {
    "서연고": ["서울대", "연세대", "고려대"],
    "고연대": ["연세대", "고려대"],
    "연고대": ["연세대", "고려대"],
    "서성한": ["서강대", "성균관대","한양대"]
}

# 조사/어미/문장부호 정리
JOSA = ["은","는","이","가","을","를","과","와","랑","도","에서","의","까지","만","는요","인가요","입니다"]
JOSA = sorted(JOSA, key=len, reverse=True)
RX_TRAIL = re.compile(r"[^\w가-힣]+$")

def strip_suffixes(s: str) -> str:
    s = RX_TRAIL.sub("", s)
    for suf in JOSA:
        if s.endswith(suf):
            s = s[:-len(suf)]
            break
    return s

def normalize_unitoken(tok: str) -> str:
    s = strip_suffixes(tok.replace(" ",""))
    # '대학교/대학' → '대' 정규화
    if s.endswith("대학교"): s = s[:-3] + "대"
    elif s.endswith("대학"):  s = s[:-2] + "대"
    return s

def match_spans(tokens: List[str]) -> List[Tuple[int,int]]:
    """
    토큰 리스트에서 '학교 사전'과 일치하는 가장 긴 비중첩 스팬 탐색.
    반환: (start, end_exclusive) 리스트
    """
    n = len(tokens)
    spans = []
    i = 0
    while i < n:
        best = None
        # 2-토큰(예: '고려' '대학교') 우선
        if i + 1 < n:
            s2 = normalize_unitoken(tokens[i]) + normalize_unitoken(tokens[i+1])
            if s2 in ALIAS_TO_CANON:
                best = (i, i+2)
        # 1-토큰(단일 축약형/복합 축약형)
        s1 = normalize_unitoken(tokens[i])
        if s1 in ALIAS_TO_CANON or s1 in COMPOSITES:
            if best is None:
                best = (i, i+1)
        if best:
            spans.append(best)
            i = best[1]
        else:
            i += 1
    return spans

def constrain_tags(tokens: List[str], _model_tags: List[str]) -> List[str]:
    """
    모델 태그와 무관하게, 사전에 일치하는 스팬만 B-UNI/I-UNI로 강제.
    - '수시' 같은 비-학교 단어는 반드시 O.
    - '연세대', '건대랑', '고려 대학교' 등은 정규화 매칭으로 B/I 부여.
    """
    n = len(tokens)
    new_tags = ["O"] * n
    for s, e in match_spans(tokens):
        new_tags[s] = "B-UNI"
        for k in range(s+1, e):
            new_tags[k] = "I-UNI"
    return new_tags
