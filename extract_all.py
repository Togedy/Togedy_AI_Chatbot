# -*- coding: utf-8 -*-
"""
extract_all.py — 로컬 utils 충돌 완전 해결판 (이 파일만 수정)

요약
- UNI: KORBERT_NER_UNI/test_uni.py 파이프라인(모델 로드 → 제약 → 후처리) 그대로 수행
  (임포트 직전 'utils/postprocess/lexicon'을 UNI 폴더 것으로 임시 바인딩)
- TYPE: KORBERT_NER_TYPE/predict.py 의 predict() 호출
  (임포트 직전 'utils'를 TYPE 폴더 것으로 임시 바인딩)
- KEYWORD: KORBERT_NER_KEYWORD/keyword_extractor.py 의 KeywordExtractor 사용
  (임포트 직전 'utils'를 KEYWORD 폴더 것으로 임시 바인딩; 필요 시)
- Gemini 재분류, 최종 분류 규칙, 시간/평균/전체, TSV 저장

최종 분류 규칙
- UNI ∧ KEYWORD → "문서탐색"
- ¬UNI ∧ KEYWORD → "재질문"
- 그 외 → "답변 생성"
"""

import argparse
import importlib
import importlib.util
import json
import os
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

THIS = os.path.dirname(os.path.abspath(__file__))
if THIS not in sys.path:
    sys.path.insert(0, THIS)

UNI_DIR = os.path.join(THIS, "KORBERT_NER_UNI")
TYPE_DIR = os.path.join(THIS, "KORBERT_NER_TYPE")
KW_DIR   = os.path.join(THIS, "KORBERT_NER_KEYWORD")

# ─────────────────────────────────────────────────────────────────────────────
# 공통 임포트 유틸
# ─────────────────────────────────────────────────────────────────────────────
def _safe_import(name: str):
    try:
        return importlib.import_module(name)
    except Exception:
        return None

def _import_by_path(modname: str, abspath: str):
    if not os.path.exists(abspath):
        return None
    spec = importlib.util.spec_from_file_location(modname, abspath)
    if not spec or not spec.loader:
        return None
    m = importlib.util.module_from_spec(spec)
    sys.modules[modname] = m
    spec.loader.exec_module(m)
    return m

class _LocalSwap:
    """
    로컬 파일을 임시로 sys.modules[name] 에 바인딩하여
    'from utils import ...' 같은 상대명 임포트 충돌을 방지.
    예) mapping = {'utils': '.../KORBERT_NER_TYPE/utils.py'}
    """
    def __init__(self, mapping: Dict[str, str]):
        self.mapping = mapping
        self._orig: Dict[str, Any] = {}

    def __enter__(self):
        for name, path in self.mapping.items():
            self._orig[name] = sys.modules.get(name)
            mod = _import_by_path(name, path)
            if mod is None:
                raise ImportError(f"로컬 모듈 로드 실패: {name} <- {path}")
        return self

    def __exit__(self, exc_type, exc, tb):
        for name in self.mapping:
            if self._orig[name] is None:
                if name in sys.modules:
                    del sys.modules[name]
            else:
                sys.modules[name] = self._orig[name]

# ─────────────────────────────────────────────────────────────────────────────
# .env
# ─────────────────────────────────────────────────────────────────────────────
def load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    return os.getenv("GOOGLE_API_KEY", ""), os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

# ─────────────────────────────────────────────────────────────────────────────
# UNI 추출기: test_uni 파이프라인 그대로
# ─────────────────────────────────────────────────────────────────────────────
class UniExtractor:
    def __init__(self, max_len: int = 128):
        self.max_len = max_len

        # test_uni.py가 'from utils/postprocess/lexicon import ...'를 쓰므로,
        # 임포트 직전 UNI 디렉터리의 파일을 이름대로 바인딩
        with _LocalSwap({
            "utils":      os.path.join(UNI_DIR, "utils.py"),
            "postprocess":os.path.join(UNI_DIR, "postprocess.py"),
            "lexicon":    os.path.join(UNI_DIR, "lexicon.py"),
        }):
            tu = _safe_import("test_uni") or _safe_import("KORBERT_NER_UNI.test_uni") \
                 or _import_by_path("test_uni", os.path.join(UNI_DIR, "test_uni.py"))
            if not tu:
                raise ImportError("KORBERT_NER_UNI/test_uni.py 를 찾을 수 없습니다.")

            self.load_model = getattr(tu, "load_model", None)
            self.LABEL_PATH = getattr(tu, "LABEL_PATH", None)
            if not callable(self.load_model) or not isinstance(self.LABEL_PATH, str):
                raise ImportError("test_uni.py에 load_model 또는 LABEL_PATH가 없습니다.")

            utils = sys.modules["utils"]
            post  = sys.modules["postprocess"]
            lexc  = sys.modules["lexicon"]

            self.get_label_list = getattr(utils, "get_label_list", None)
            self.postprocess    = getattr(post,  "postprocess_ner_output", None)
            self.constrain_tags = getattr(lexc,  "constrain_tags", None)
            if not (callable(self.get_label_list) and callable(self.postprocess) and callable(self.constrain_tags)):
                raise ImportError("UNI: get_label_list/postprocess_ner_output/constrain_tags 확인 필요.")

        # 모델/라벨 로드(1회)
        self.tokenizer, self.model = self.load_model()
        labels, label_to_id, id_to_label = self.get_label_list(self.LABEL_PATH)
        self.id2lab = {i: l for i, l in enumerate(labels)}

        import torch  # noqa: F401
        self._torch = __import__("torch")

    def extract_uni(self, text: str) -> List[str]:
        words = text.split()
        tk = self.tokenizer
        cls_id = tk.cls_token_id or tk.convert_tokens_to_ids("[CLS]")
        sep_id = tk.sep_token_id or tk.convert_tokens_to_ids("[SEP]")
        pad_id = tk.pad_token_id or tk.convert_tokens_to_ids("[PAD]")

        wp_ids = [cls_id]
        word_first_wp = []
        for w in words:
            wp = tk.tokenize(w) or ["[UNK]"]
            word_first_wp.append(len(wp_ids))
            wp_ids += tk.convert_tokens_to_ids(wp)
        wp_ids += [sep_id]
        attention = [1] * len(wp_ids)

        if len(wp_ids) < self.max_len:
            pad_len = self.max_len - len(wp_ids)
            wp_ids += [pad_id] * pad_len
            attention += [0] * pad_len
        else:
            wp_ids = wp_ids[:self.max_len]
            attention = attention[:self.max_len]

        torch = self._torch
        input_ids = torch.tensor([wp_ids])
        attention_mask = torch.tensor([attention])

        self.model.eval()
        with torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        preds = out.logits[0].argmax(-1).tolist()

        per_word = [self.id2lab[preds[idx]] for idx in word_first_wp]
        per_word = self.constrain_tags(words, per_word)
        result = self.postprocess(words, per_word)
        return list(result.get("UNI") or [])

# ─────────────────────────────────────────────────────────────────────────────
# TYPE 추출기: predict.py (로컬 utils 충돌 방지)
# ─────────────────────────────────────────────────────────────────────────────
class TypeExtractor:
    def __init__(self):
        self._predict = None
        # 임포트 직전 TYPE 디렉터리의 utils를 바인딩하여 load_label_list 충돌 방지
        with _LocalSwap({"utils": os.path.join(TYPE_DIR, "utils.py")}):
            mod = _safe_import("predict") or _safe_import("KORBERT_NER_TYPE.predict") \
                  or _import_by_path("predict", os.path.join(TYPE_DIR, "predict.py"))
            if mod:
                self._predict = getattr(mod, "predict", None)

    def extract_type(self, text: str) -> List[str]:
        if callable(self._predict):
            try:
                out = self._predict(text)
                if isinstance(out, dict):
                    typ = out.get("TYPE", [])
                    if isinstance(typ, str):
                        return [typ] if typ else []
                    return list(typ) if typ else []
            except Exception:
                return []
        return []

# ─────────────────────────────────────────────────────────────────────────────
# KEYWORD 추출기: keyword_extractor.py (로컬 utils 충돌 방지)
# ─────────────────────────────────────────────────────────────────────────────
class KeywordExtractorBridge:
    def __init__(self, topn: int = 10):
        self.topn = topn
        self.ke = None
        # 임포트 직전 KEYWORD 디렉터리의 utils를 바인딩 (있을 때만)
        utils_path = os.path.join(KW_DIR, "utils.py")
        mapping = {"utils": utils_path} if os.path.exists(utils_path) else {}
        with _LocalSwap(mapping) if mapping else _NullCtx():
            mod = _safe_import("KORBERT_NER_KEYWORD.keyword_extractor") \
                  or _safe_import("keyword_extractor") \
                  or _import_by_path("keyword_extractor", os.path.join(KW_DIR, "keyword_extractor.py"))
            if mod:
                try:
                    self.ke = getattr(mod, "KeywordExtractor")(use_model=True)
                except Exception:
                    try:
                        self.ke = getattr(mod, "KeywordExtractor")()
                    except Exception:
                        self.ke = None

    def extract_keywords(self, text: str) -> List[str]:
        if not self.ke:
            return []
        try:
            return list(self.ke.extract(text, topn=self.topn, allow_composite=True))
        except Exception:
            try:
                return list(self.ke.extract_model_only(text, topn=self.topn))
            except Exception:
                return []

class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False

# ─────────────────────────────────────────────────────────────────────────────
# Gemini 재분류/정렬
# ─────────────────────────────────────────────────────────────────────────────
def gemini_sort(api_key: str, model: str, uni: List[str], typ_list: List[str], keywords: List[str]) -> List[Dict[str, Any]]:
    typ = typ_list[0] if typ_list else ""
    default_pairs = [{"UNI": u, "TYPE": typ, "KEYWORD": keywords[:]} for u in (uni or [None])]
    if not api_key:
        return default_pairs
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        prompt = (
            "아래 추출 결과를 대학별로 정규화해 한 줄에 하나씩 JSON 객체로 재정렬해 주세요.\n"
            "출력은 반드시 JSON Lines 형식만. 설명/코드블록 금지.\n\n"
            f"UNI 리스트: {uni}\nTYPE 후보: {typ_list}\nKEYWORD 리스트: {keywords}\n\n"
            '예:\n{"UNI":"연세대","TYPE":"수시","KEYWORD":["컴퓨터공학부","모집일정"]}\n'
            '{"UNI":"고려대","TYPE":"수시","KEYWORD":["컴퓨터공학부","모집일정"]}\n'
            "규칙:\n- UNI가 여러 개면 각 UNI마다 한 줄씩.\n- KEYWORD는 배열 유지.\n"
            "- TYPE은 가장 적합한 하나만.\n- 값 없으면 빈 문자열/배열."
        )
        resp = genai.GenerativeModel(model).generate_content(prompt)
        text = (resp.text or "").strip()
        if not text:
            return default_pairs
        out = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                obj.setdefault("UNI", None)
                obj.setdefault("TYPE", typ)
                ks = obj.get("KEYWORD", keywords[:])
                if not isinstance(ks, list):
                    ks = [ks] if ks else []
                obj["KEYWORD"] = ks
                out.append(obj)
            except Exception:
                return default_pairs
        return out or default_pairs
    except Exception:
        return default_pairs

# ─────────────────────────────────────────────────────────────────────────────
# 최종 분류 규칙
# ─────────────────────────────────────────────────────────────────────────────
def final_bucket(uni: List[str], keywords: List[str]) -> str:
    if uni and keywords:
        return "문서탐색"
    if (not uni) and keywords:
        return "재질문"
    return "답변 생성"

# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────
def read_questions(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {path}")
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            out.append(line)
    return out

def write_results(path: str, rows: List[Tuple[str, List[str], List[str], List[str], str, float]]):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("query\tuni\ttype\tkeyword\tfinal_bucket\telapsed_sec\n")
        for q, uni, typ, kwd, bucket, sec in rows:
            f.write(f"{q}\t{'|'.join(uni)}\t{'|'.join(typ)}\t{'|'.join(kwd)}\t{bucket}\t{sec:.3f}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 한 문장 처리
# ─────────────────────────────────────────────────────────────────────────────
def process_sentence(text: str, uni_ex: UniExtractor, type_ex: TypeExtractor, kw_ex: KeywordExtractorBridge,
                     api_key: str, model: str) -> Dict[str, Any]:
    t0 = time.perf_counter()
    uni = uni_ex.extract_uni(text)
    typ = type_ex.extract_type(text)
    kwd = kw_ex.extract_keywords(text)
    pairs = gemini_sort(api_key, model, uni, typ, kwd)
    bucket = final_bucket(uni, kwd)
    elapsed = time.perf_counter() - t0
    return {
        "text": text,
        "extracted": {"UNI": uni, "TYPE": typ, "KEYWORD": kwd},
        "gemini_sorted": pairs,
        "decision": bucket,
        "elapsed_sec": elapsed,
    }

# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", default="test.txt")
    ap.add_argument("--output", "-o", default="results.tsv")
    ap.add_argument("--topn", type=int, default=10)
    args = ap.parse_args()

    api_key, gemini_model = load_env()

    uni_ex = UniExtractor(max_len=128)
    type_ex = TypeExtractor()
    kw_ex   = KeywordExtractorBridge(topn=args.topn)

    qs = read_questions(args.input)
    if not qs:
        print("[안내] 입력 문장이 없습니다.")
        return

    print(f"총 {len(qs)}개의 문장 처리 시작...\n")
    rows, per_line = [], []
    t0 = time.perf_counter()
    width = len(str(len(qs)))

    for idx, q in enumerate(qs, 1):
        res = process_sentence(q, uni_ex, type_ex, kw_ex, api_key, gemini_model)
        per_line.append(res["elapsed_sec"])
        e = res["extracted"]

        print("-" * 78)
        print(f"[{idx:>{width}}/{len(qs)}] Q: {res['text']}")
        print(f" - 추출 NER → UNI: {e['UNI']} / TYPE: {e['TYPE']} / KEYWORD: {e['KEYWORD']}")
        print(" - Gemini 재분류 결과:")
        for p in res["gemini_sorted"]:
            uni = p.get("UNI")
            typ = p.get("TYPE")
            kws = p.get("KEYWORD")
            if not isinstance(kws, list):
                kws = [kws] if kws else []
            print(f"   (UNI : {uni} / 타입 : {typ} / 키워드 : {kws})")
        print(f" - 최종 분류: {res['decision']}")
        print(f" - 처리 시간: {res['elapsed_sec']:.3f}초")

        rows.append((q, e["UNI"], e["TYPE"], e["KEYWORD"], res["decision"], res["elapsed_sec"]))

    total = time.perf_counter() - t0
    avg = statistics.mean(per_line) if per_line else 0.0

    write_results(args.output, rows)
    print("-" * 78)
    print(f"문장 수: {len(per_line)}")
    print(f"평균 처리 시간: {avg:.3f}초")
    print(f"전체 처리 시간: {total:.3f}초")
    print(f"결과 저장: {os.path.abspath(args.output)}")
    print("-" * 78)

if __name__ == "__main__":
    main()
