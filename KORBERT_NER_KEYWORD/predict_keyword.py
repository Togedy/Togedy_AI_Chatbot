# -*- coding: utf-8 -*-
"""
predict_keyword.py — KEYWORD(B/I/O) 예측 + 후처리(전공/속성 분리)
- 모델의 config.id2label 우선 사용(라벨 파일 없어도 동작)
- BIO 교정
- 전공/키워드 사전 매칭 + N-그램 인식
- 금칙어 억제(누가/누구/누/더 등)
- 허용 토큰(전공/키워드/헤드/접미사) 외 라벨 O 강등
- ✅ 전공과 속성을 '서로 다른' 스팬으로 유지(자동 병합/확장 제거)
- 스팬 병합 시 '명사 위주'만 포함(koNLPy Okt 있으면 POS, 없으면 휴리스틱)

직접 실행 예:
python predict_keyword.py
"""

from typing import List, Dict, Tuple, Optional
import os
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

LABEL_PATH = "./KORBERT_NER_KEYWORD/data/label.txt"
MODEL_DIR  = "./KORBERT_NER_KEYWORD/results_keyword"

# (선택) KoNLPy(Okt) 사용
_OKT = None
try:
    from konlpy.tag import Okt
    _OKT = Okt()
except Exception:
    _OKT = None

# 안전한 조사 목록(과/와/은/는/이/가/을/를 등)
_JOSA_LIST = (
    "으로", "로", "에서", "에게", "에게서", "에게는",
    "과", "와", "은", "는", "이", "가", "을", "를",
    "도", "만", "의", "랑", "하고", "보다", "부터", "까지", "에"
)

# 전공/키워드 사전(정규화 비교: 조사 1회 제거 후)
MAJOR_SYNS = {
    "컴퓨터공학부", "컴퓨터 공학부", "컴공",
    "경제학부", "경제 과", "경제과",
    "경영학과", "경영 과", "경영과",
    "전자공학부", "전자 공학부", "전자과",
    "ai학과", "인공지능 학과", "ai 과",
    "데이터사이언스학부", "데이터 사이언스 학부", "데사학부",
    "수학과", "물리학과", "화학과",
    "생명과학과", "생명 과학 과",
}

# ✅ 단일 키워드(‘입결’ 추가)
KEY_UNIGRAMS = {
    "모집일정", "모집인원", "경쟁률", "수능최저", "등록기간",
    "합격자", "합격기준", "충원율", "원서접수", "전형", "일정", "서류평가",
    "입결",
}
# 복합 키워드(토큰 시퀀스)
KEY_NGRAMS = [
    ["전형", "일정"],
    ["모집", "일정"],
    ["원서", "접수", "기간"],
    ["수능", "최저"],
    ["제출", "서류"],
    ["모집", "인원"],
]
# 합성형(연결된 표기)
KEY_COMPOSITES = {
    "전형일정", "모집일정", "원서접수기간", "수능최저",
    "모집인원", "등록기간", "서류평가", "합격기준",
}

_MAJOR_SUFFIXES = ("학과", "학부")

# 키워드 속성(헤드)
KEY_HEADS = {
    "모집", "모집일정", "모집인원", "전형", "일정",
    "원서", "접수", "기간", "경쟁률", "수능", "최저",
    "등록", "합격자", "합격", "충원", "서류", "평가", "입결",
}

# 금칙어(정규화 포함) — 무조건 O로
BANNED_NORMS = {"누", "누가", "누구", "더"}

def _flatten_ngram_set(ngrams):
    s = set()
    for ng in ngrams:
        s.update(ng)
    return s

# 허용 토큰(정규화 기준)
ALLOWED_NORMS = (
    set(MAJOR_SYNS)
    | set(KEY_UNIGRAMS)
    | set(KEY_COMPOSITES)
    | set(KEY_HEADS)
    | _flatten_ngram_set(KEY_NGRAMS)
)

# ---------------- 공통 유틸 ----------------
def _strip_one_josa(word: str) -> str:
    """끝 조사 1개 제거(있으면)."""
    for j in sorted(_JOSA_LIST, key=len, reverse=True):
        if word.endswith(j) and len(word) > len(j):
            return word[: -len(j)]
    return word

def _normalize_tokens(words: List[str]) -> List[str]:
    return [_strip_one_josa(w) for w in words]

def repair_bio(tags: List[str]) -> List[str]:
    """I가 단독/시작이면 B로 교정."""
    fixed, prev = [], "O"
    for t in tags:
        if t == "I-KEYWORD" and prev not in ("B-KEYWORD", "I-KEYWORD"):
            t = "B-KEYWORD"
        fixed.append(t); prev = t
    return fixed

def _mark_span(tags: List[str], i: int, L: int):
    tags[i] = "B-KEYWORD"
    for j in range(1, L):
        tags[i + j] = "I-KEYWORD"

# ---------------- 사전/규칙 보강 ----------------
def apply_gazetteer(words: List[str], tags: List[str]) -> List[str]:
    norms = _normalize_tokens(words)
    n = len(words)
    # 1) 단일 매칭
    for i in range(n):
        w = norms[i]
        if w in MAJOR_SYNS or w in KEY_UNIGRAMS or w in KEY_COMPOSITES:
            tags[i] = "B-KEYWORD"
    # 2) 복합 키워드
    for ngram in KEY_NGRAMS:
        L = len(ngram)
        for i in range(0, n - L + 1):
            if norms[i:i+L] == ngram:
                _mark_span(tags, i, L)
    # 3) 금칙어 강등
    for i in range(n):
        if norms[i] in BANNED_NORMS:
            tags[i] = "O"
    return repair_bio(tags)

def restrict_to_allowed_tokens(words: List[str], tags: List[str]) -> List[str]:
    """허용 목록(ALLOWED_NORMS/접미사) 외 라벨은 O로 강등."""
    norms = _normalize_tokens(words)
    for i, t in enumerate(tags):
        if t == "O":
            continue
        w = norms[i]
        ok = (w in ALLOWED_NORMS) or any(w.endswith(suf) for suf in _MAJOR_SUFFIXES)
        if not ok:
            tags[i] = "O"
    return repair_bio(tags)

# ---------------- 명사성 판별 ----------------
def _is_nounish_token(token: str) -> bool:
    norm = _strip_one_josa(token)
    # KoNLPy 사용 시
    if _OKT is not None:
        try:
            pos = _OKT.pos(norm, norm=True, stem=True)
            if any(t == "Noun" for _, t in pos):
                return True
        except Exception:
            pass
    # 휴리스틱: 허용 목록/접미사/전형적 명사 접미사
    if norm in ALLOWED_NORMS or any(norm.endswith(s) for s in _MAJOR_SUFFIXES):
        return True
    noun_like_suffix = ("일정", "인원", "경쟁률", "기간", "평가", "요소", "자격", "최저", "서류", "발표", "기준", "입결")
    if any(norm.endswith(s) for s in noun_like_suffix):
        return True
    if len(norm) == 1:
        return False
    return False

def _merge_bio_nouns_only(words: List[str], tags: List[str],
                          b_tag="B-KEYWORD", i_tag="I-KEYWORD",
                          join_with_space: bool = False) -> List[str]:
    """BIO 스팬 병합(명사성 토큰만 포함)."""
    spans, cur = [], []
    for w, t in zip(words, tags):
        if t == b_tag:
            if cur:
                spans.append(" ".join(cur) if join_with_space else "".join(cur))
                cur = []
            nw = _strip_one_josa(w)
            if _is_nounish_token(nw):
                cur.append(nw)
            else:
                cur = []
        elif t == i_tag:
            if cur:
                nw = _strip_one_josa(w)
                if _is_nounish_token(nw):
                    cur.append(nw)
                # 비명사는 건너뜀
            else:
                nw = _strip_one_josa(w)
                if _is_nounish_token(nw):
                    cur = [nw]
        else:
            if cur:
                spans.append(" ".join(cur) if join_with_space else "".join(cur))
                cur = []
    if cur:
        spans.append(" ".join(cur) if join_with_space else "".join(cur))
    return spans

# ---------------- 인코딩/라벨 매핑 ----------------
def _encode_words(tok: AutoTokenizer, words: List[str]):
    enc = tok(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )
    enc.pop("token_type_ids", None)
    try:
        word_ids = enc.word_ids()
    except TypeError:
        word_ids = enc.word_ids(batch_index=0)
    return enc, word_ids

def _maybe_load_id2label_from_file(path: str) -> Optional[Dict[int, str]]:
    """label.txt가 있으면 사용(O/B-KEYWORD/I-KEYWORD 가정)."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            labels = [ln.strip() for ln in f if ln.strip()]
        return {i: lab for i, lab in enumerate(labels)}
    except Exception:
        return None

def _get_id2label(model) -> Dict[int, str]:
    """모델 config.id2label을 우선 사용, 없으면 label.txt, 최후엔 기본값."""
    id2label = getattr(model.config, "id2label", None)
    if isinstance(id2label, dict) and id2label:
        try:
            return {int(k): v for k, v in id2label.items()}
        except Exception:
            return {k: v for k, v in id2label.items()}
    from_file = _maybe_load_id2label_from_file(LABEL_PATH)
    if from_file:
        return from_file
    return {0: "O", 1: "B-KEYWORD", 2: "I-KEYWORD"}

# ---------------- 메인 예측 ----------------
def predict(text: str,
            model_dir: str = MODEL_DIR,
            use_gpu: bool = True,
            join_with_space: bool = False) -> Dict[str, List[str]]:
    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"

    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir).to(device)
    model.eval()
    id2label = _get_id2label(model)

    words = text.strip().split()
    if not words:
        return {"tokens": [], "tags": [], "KEYWORD": []}

    # 1) 모델 예측
    enc, word_ids = _encode_words(tok, words)
    with torch.no_grad():
        for k in enc:
            enc[k] = enc[k].to(device)
        logits = model(**enc).logits
        pred_ids = torch.argmax(logits, dim=-1).squeeze(0).tolist()

    # 단어 단위 정렬(첫 서브토큰 라벨)
    tags_word, seen = [], set()
    for wid, pid in zip(word_ids, pred_ids):
        if wid is None:
            continue
        if wid not in seen:
            tags_word.append(id2label.get(pid, "O")); seen.add(wid)

    # 길이 보정
    if len(tags_word) > len(words):
        tags_word = tags_word[:len(words)]
    elif len(tags_word) < len(words):
        tags_word += ["O"] * (len(words) - len(tags_word))

    # 2) BIO 교정
    tags_word = repair_bio(tags_word)
    # 3) 사전 보강 + 금칙어 강등
    tags_word = apply_gazetteer(words, tags_word)
    # 4) 허용 토큰 외 라벨 제거
    tags_word = restrict_to_allowed_tokens(words, tags_word)

    # ✅ 전공+속성 자동 병합/확장 단계는 제거하여
    #    전공과 속성이 각기 별도 스팬으로 유지되도록 함.
    # (merge_major_head_pairs / maybe_extend_with_major 미사용)

    # 5) 명사 위주 병합
    keywords = _merge_bio_nouns_only(words, tags_word, "B-KEYWORD", "I-KEYWORD", join_with_space)

    return {"tokens": words, "tags": tags_word, "KEYWORD": keywords}

# ---------------- 수동 테스트 ----------------
if __name__ == "__main__":
    samples = [
        "컴퓨터공학부 모집인원 알려줘",         # → ['컴퓨터공학부', '모집인원']
        "경제학부 전형 일정",                   # → ['경제학부', '전형일정'] (또는 '전형 일정' join_with_space=True)
        "컴퓨터공학부와 경제학부 모두 모집일정 궁금해",  # → ['컴퓨터공학부','경제학부','모집일정']
        "모집일정과 전공명을 함께 말했어",       # → ['모집일정']
        "건대랑 서연고 수시 모집 일정 알려줘",   # → ['모집일정']
        "건국대랑 연세대랑 붙으면 누가 이겨?",   # → []
        "서성한에서 입결 누가 더 높아?",        # → ['입결']
        "연세대 출신 연예인 누구 있어?"          # → []
    ]
    for s in samples:
        out = predict(s)
        print("Sentence:", s)
        print("Tokens:", out["tokens"])
        print("Tags:  ", out["tags"])
        print("KEYWORD:", out["KEYWORD"])
        print()
