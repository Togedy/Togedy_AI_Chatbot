# -*- coding: utf-8 -*-
"""
keyword_extractor.py
- NER 모델(학습은 모델로만) 결과를 '우선' 사용하고,
  문장 안에서만 동작하는 '가벼운 규칙'으로 '보강만' 수행.
- PDF 사용❌, 부분문자열 확장❌, 문장 외 생성❌

정책:
  1) 우선 NER 결과(KEYWORD)를 그대로 채택 (가능한 한 가공 최소화)
  2) 규칙은 누락 보강만:
     - 붙여/띄어 혼용 합성(문장에 두 파트가 '모두' 있을 때만): 전형+일정→전형일정, 일반+학생→일반학생, 모집+일정→모집일정 등
     - 단과/학부/학과 계열: '경제학부', '컴퓨터공학부' 등은 그대로 유지하되, '학부','학과' 단독은 가능한 한 제거
     - 너무 일반 단어(기준/안내)는 보다 구체 키워드가 있으면 제거
  3) 학교명/전형타입(정시/수시) 토큰은 최종에서 제거
  4) 합성어가 생기면 분해형(전형/일정 등)은 제거
"""

from __future__ import annotations
from typing import List, Iterable, Optional, Tuple, Set
import re

# -------------------- 유틸 --------------------
_nonword = re.compile(r"[^가-힣A-Za-z0-9\s]")
_space = re.compile(r"\s+")

def normalize_text(s: str) -> str:
    return _space.sub(" ", _nonword.sub(" ", s)).strip().lower()

def split_tokens(s: str) -> List[str]:
    # 간단 토큰: 자모/영숫자 연속
    return re.findall(r"[가-힣A-Za-z0-9]+", s)

def uniq(seq: Iterable[str]) -> List[str]:
    out, seen = [], set()
    for x in seq:
        if not x: continue
        if x not in seen:
            seen.add(x); out.append(x)
    return out

# -------------------- 금칙/정책 --------------------
SCHOOL_SYNS = {
    "건국대학교": {"건국대","건국대학교","건대","konkuk","ku"},
    "연세대학교": {"연세대","연세대학교","yonsei"},
    "고려대학교": {"고려대","고려대학교"},
    "서울대학교": {"서울대","서울대학교","snu"},
}
TYPE_SYNS_FLAT = {"정시","수시","정시모집","수시모집"}

SCHOOL_SYNS_FLAT: Set[str] = set()
for _, syns in SCHOOL_SYNS.items():
    for s in syns:
        SCHOOL_SYNS_FLAT.add(normalize_text(s))

def is_school_or_type(token: str) -> bool:
    n = normalize_text(token)
    return (n in SCHOOL_SYNS_FLAT) or (n in TYPE_SYNS_FLAT)

# 합성 후보(문장 안에 두 파트가 '모두' 있으면 붙여 만듦)
JOINABLE: Set[Tuple[str, str]] = {
    ("전형","일정"),
    ("모집","일정"),
    ("원서","접수"), ("접수","기간"),
    ("모집","인원"), ("모집","단위"),
    ("수능","최저"),
    ("영어","등급"), ("등급","환산"),
    ("전형","방법"),
    ("일반","학생"),  # KU일반학생 보강
}

# 접미사(명사성) — 규칙 보강 시 허용
NOUN_SUFFIX = ("학과","학부","전공","모집군","모집단위","정원","일정","인원","경쟁률","기간",
               "평가","기준","최저","서류","요강","입결","반영","환산","등급","비율","방법","영역","과목","가산점","수능","군")

# 너무 일반적(합성/구체 키워드가 함께 있으면 제거)
GENERIC_STOP = {"기준","안내"}

def _is_clean(k: str) -> bool:
    if len(k) < 2 or len(k) > 20: return False
    if any(ch.isdigit() for ch in k): return False
    return True

# -------------------- 핵심 클래스 --------------------
class KeywordExtractor:
    def __init__(self, use_model: bool = True):
        self.use_model = use_model
        self._model_predict = None
        if use_model:
            # 안전 임포트: 절대 → 상대 → 로컬
            try:
                from KORBERT_NER_KEYWORD.predict_keyword import predict as _pred
                self._model_predict = _pred
            except Exception:
                try:
                    from .predict_keyword import predict as _pred  # type: ignore
                    self._model_predict = _pred
                except Exception:
                    try:
                        from predict_keyword import predict as _pred  # type: ignore
                        self._model_predict = _pred
                    except Exception:
                        self._model_predict = None

    # --------- 1) NER 결과 우선 ---------
    def _model_keywords(self, text: str) -> List[str]:
        if self._model_predict is None:
            return []
        try:
            out = self._model_predict(text)       # {"KEYWORD": [...]} 형태 가정
            kws = out.get("KEYWORD") or []
            # NER 결과는 가급적 '원형' 유지 (과결합 분해 X)
            # 단, 학교/타입, 노이즈 필터만 적용
            cleaned = []
            for k in kws:
                if not _is_clean(k): continue
                if is_school_or_type(k): continue
                cleaned.append(k)
            return uniq(cleaned)
        except Exception:
            return []

    # --------- 2) 규칙 보강(문장 안에서만) ---------
    def _rule_boost(self, text: str, model_kws: List[str]) -> List[str]:
        low = normalize_text(text)
        toks = split_tokens(low)
        cand: List[str] = []

        # (a) 합성어: 문장 안에 두 파트가 모두 있으면 생성 (JOINABLE)
        tokset = set(toks)
        for a,b in JOINABLE:
            # 1) 공백 토큰 기준
            if (a in tokset) and (b in tokset):
                cand.append(a+b); continue
            # 2) 문장 내 연속표기(띄어쓰기/조사 변형) 허용: 'a\s*?b'
            if re.search(rf"{a}\s*{b}", low):
                cand.append(a+b)

        # (b) 전공/학부/학과류: 토큰 중 접미사 매칭
        for t in toks:
            if any(t.endswith(suf) for suf in NOUN_SUFFIX):
                cand.append(t)

        # (c) 너무 일반 단어 정리: '기준','안내'는 다른 구체 키워드가 있으면 제거
        result = uniq([k for k in cand if _is_clean(k) and not is_school_or_type(k)])
        if any(x not in GENERIC_STOP for x in (model_kws + result)):
            result = [x for x in result if x not in GENERIC_STOP]

        # (d) 이미 모델이 잡은 것은 그대로 두고, 없는 것만 보강
        add_only = [k for k in result if k not in set(model_kws)]
        return add_only

    # --------- 3) 합성 있으면 분해형 제거 ---------
    def _dedup_splits(self, kws: List[str]) -> List[str]:
        present = set(kws)
        remove = set()
        # JOINABLE 사전에 등장하는 분해형만 제거 (과도한 삭제 방지)
        for a,b in JOINABLE:
            comp = a+b
            if comp in present:
                if a in present: remove.add(a)
                if b in present: remove.add(b)
        # '학부','학과' 단독 제거(구체 명사가 있으면)
        if ("학부" in present) and any(x.endswith("학부") and len(x) > 2 for x in present):
            remove.add("학부")
        if ("학과" in present) and any(x.endswith("학과") and len(x) > 2 for x in present):
            remove.add("학과")
        return [k for k in kws if k not in remove]

    # --------- 공개 API ---------
    def extract(self, text: str, topn: Optional[int] = None, allow_composite: bool = True) -> List[str]:
        # 1) 모델 우선
        model_kws = self._model_keywords(text)

        # 2) 규칙 보강(문장에서만, 모델이 놓친 것만 추가)
        rule_add = self._rule_boost(text, model_kws) if allow_composite else []
        merged = uniq(model_kws + rule_add)

        # 3) 합성 존재 시 분해형 제거
        deduped = self._dedup_splits(merged)

        # 4) 최종 상위 N
        return deduped[:topn] if topn else deduped

    # (옵션) 순수 모델 결과만 보고 싶을 때
    def extract_model_only(self, text: str, topn: Optional[int] = None) -> List[str]:
        out = self._model_keywords(text)
        return out[:topn] if topn else out


# -------------- 데모 --------------
if __name__ == "__main__":
    ke = KeywordExtractor(use_model=True)
    samples = [
        "건대 정시 전형 일정 알려줘",
        "건국대 정시 모집인원",
        "KU일반학생 전형 방법",
        "영어등급 환산표 보여줘",
        "가군 모집단위 안내",
        "수능 최저 기준은?",
        "입결 궁금해",
        "컴퓨터공학부 모집인원 알려줘",
        "경제학부 전형 일정",
        "컴퓨터공학부와 경제학부 모두 모집일정 궁금해",
    ]
    for s in samples:
        print(f"\nQ: {s}")
        print("->", ke.extract(s, topn=10))
