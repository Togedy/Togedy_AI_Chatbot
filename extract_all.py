# -*- coding: utf-8 -*-
"""
Togedy_AI_Train/extract_all.py

기능:
1) test.txt 질문을 줄단위로 읽음(#, //, 빈 줄 무시)
2) 학습데이터(train.tsv, eval.tsv)에서 BIO 태그 기반 phrase lexicon 생성
3) 모델 예측 결과(test_keyword_extractor.py, test.py, test_uni.py 호출)
4) 모델 결과 + 학습데이터 사전 기반 매칭 병합
5) 각 문장별 추론 시간(ms)과 평균 시간 출력
6) 결과를 results.tsv로 저장
"""

import argparse
import importlib
import os
import sys
import time
import statistics
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

# ------------------------------------------------------------
# 안전 임포트 및 callable 탐색
# ------------------------------------------------------------
def _safe_import(module_name: str) -> Optional[Any]:
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def _find_callable(mod: Any, names: List[str]):
    if not mod:
        return None
    for n in names:
        fn = getattr(mod, n, None)
        if callable(fn):
            return fn
    return None


# ------------------------------------------------------------
# BIO TSV에서 phrase lexicon 생성
# ------------------------------------------------------------
def load_phrases_from_bio(tsv_paths: List[str], target_prefixes: List[str]) -> Counter:
    phrases = Counter()
    for path in tsv_paths:
        if not path or not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            sent_tokens, sent_labels = [], []

            def _flush_sentence():
                if not sent_tokens:
                    return
                n = len(sent_tokens)
                i = 0
                while i < n:
                    label = sent_labels[i] if i < n else "O"
                    if label.startswith("B-"):
                        ent = label[2:]
                        if ent in target_prefixes:
                            j = i + 1
                            while j < n and sent_labels[j] == f"I-{ent}":
                                j += 1
                            phrase = "".join(sent_tokens[i:j])
                            if phrase:
                                phrases[(ent, phrase)] += 1
                            i = j
                            continue
                    i += 1

            for raw in f:
                line = raw.strip()
                if not line:
                    _flush_sentence()
                    sent_tokens, sent_labels = [], []
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    token, label = parts[0], parts[1]
                else:
                    token, label = parts[0], "O"
                sent_tokens.append(token)
                sent_labels.append(label)
            _flush_sentence()
    return phrases


# ------------------------------------------------------------
# 사전 매칭 함수
# ------------------------------------------------------------
def lexicon_match(text: str, cand_phrases: List[str]) -> List[str]:
    if not text or not cand_phrases:
        return []
    out, seen = [], set()
    for p in sorted(set(cand_phrases), key=lambda x: (-len(x), x)):
        if p and p in text and p not in seen:
            out.append(p)
            seen.add(p)
    return out


# ------------------------------------------------------------
# 통합 추출기
# ------------------------------------------------------------
class AllExtractor:
    def __init__(
        self,
        topn_keyword: int = 10,
        uni_tsvs: Optional[List[str]] = None,
        type_tsvs: Optional[List[str]] = None,
        kw_tsvs: Optional[List[str]] = None,
    ):
        self.topn = topn_keyword

        # 모듈 로드
        self.keyword_mod = (
            _safe_import("test_keyword_extractor")
            or _safe_import("KORBERT_NER_KEYWORD.keyword_extractor")
        )
        self.type_mod = _safe_import("test") or _safe_import("type_extractor")
        self.uni_mod = _safe_import("test_uni") or _safe_import("uni_extractor")

        # keyword
        self.keyword_cls = None
        self.keyword_fn = None
        if self.keyword_mod:
            KE = getattr(self.keyword_mod, "KeywordExtractor", None)
            if KE:
                try:
                    self.keyword_cls = KE(use_model=True)
                except Exception:
                    try:
                        self.keyword_cls = KE()
                    except Exception:
                        self.keyword_cls = None
            self.keyword_fn = _find_callable(
                self.keyword_mod, ["extract", "extract_keywords", "predict"]
            )

        # type
        self.type_fn = _find_callable(
            self.type_mod, ["predict", "extract_type", "infer"]
        )

        # uni
        self.uni_fn = _find_callable(
            self.uni_mod, ["predict", "extract_uni", "infer", "predict_sentence"]
        )

        # ------------------------------------------------
        # 학습 데이터에서 phrase lexicon 생성
        # ------------------------------------------------
        default_uni = [
            "KORBERT_NER_UNI/data/train.tsv",
            "KORBERT_NER_UNI/data/eval_test.tsv",
        ]
        default_type = [
            "KORBERT_NER_TYPE/data/train.tsv",
            "KORBERT_NER_TYPE/data/eval.tsv",
        ]
        default_kw = [
            "KORBERT_NER_KEYWORD/data/train.tsv",
            "KORBERT_NER_KEYWORD/data/eval.tsv",
        ]

        def fix_paths(paths):
            root = os.path.dirname(os.path.abspath(__file__))
            fixed = []
            for p in paths:
                if not p:
                    continue
                abs_p = p if os.path.isabs(p) else os.path.join(root, p)
                fixed.append(abs_p)
            return fixed

        self.uni_phr = load_phrases_from_bio(
            fix_paths(uni_tsvs or default_uni), ["UNI"]
        )
        self.type_phr = load_phrases_from_bio(
            fix_paths(type_tsvs or default_type), ["TYPE"]
        )
        self.kw_phr = load_phrases_from_bio(
            fix_paths(kw_tsvs or default_kw), ["KEYWORD", "KW", "KEY"]
        )

        self.uni_lex = [p for (ent, p), _ in self.uni_phr.items() if ent == "UNI"]
        self.type_lex = [p for (ent, p), _ in self.type_phr.items() if ent == "TYPE"]
        self.kw_lex = [p for (ent, p), _ in self.kw_phr.items()]

    # ---------- 모델 예측 ----------
    def _predict_keywords(self, text: str) -> List[str]:
        if self.keyword_cls:
            try:
                return list(
                    self.keyword_cls.extract(text, topn=self.topn, allow_composite=True)
                )
            except Exception:
                pass
        if self.keyword_fn:
            try:
                res = self.keyword_fn(text)
                if isinstance(res, dict) and "KEYWORD" in res:
                    return res["KEYWORD"]
                if isinstance(res, list):
                    return res
            except Exception:
                pass
        return []

    def _predict_type(self, text: str) -> List[str]:
        if not self.type_fn:
            return []
        try:
            res = self.type_fn(text)
            if isinstance(res, dict):
                return res.get("TYPE", []) or res.get("type", [])
            if isinstance(res, list):
                return res
        except Exception:
            pass
        return []

    def _predict_uni(self, text: str) -> List[str]:
        if not self.uni_fn:
            return []
        try:
            res = self.uni_fn(text)
            if isinstance(res, dict):
                return res.get("UNI", []) or res.get("uni", [])
            if isinstance(res, list):
                return res
        except Exception:
            pass
        return []

    # ---------- 사전 기반 ----------
    def _lex_keywords(self, text: str) -> List[str]:
        return lexicon_match(text, self.kw_lex)

    def _lex_type(self, text: str) -> List[str]:
        return lexicon_match(text, self.type_lex)

    def _lex_uni(self, text: str) -> List[str]:
        return lexicon_match(text, self.uni_lex)

    # ---------- 병합 ----------
    def _merge(self, primary: List[str], secondary: List[str]) -> List[str]:
        seen, out = set(), []
        for lst in [primary, secondary]:
            for x in lst:
                if x and x not in seen:
                    out.append(x)
                    seen.add(x)
        return out

    # ---------- 통합 ----------
    def extract(self, text: str) -> Dict[str, List[str]]:
        m_kw, m_ty, m_un = (
            self._predict_keywords(text),
            self._predict_type(text),
            self._predict_uni(text),
        )
        d_kw, d_ty, d_un = (
            self._lex_keywords(text),
            self._lex_type(text),
            self._lex_uni(text),
        )
        return {
            "uni": self._merge(m_un, d_un),
            "type": self._merge(m_ty, d_ty),
            "keyword": self._merge(m_kw, d_kw),
        }


# ------------------------------------------------------------
# I/O 함수
# ------------------------------------------------------------
def read_questions(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {path}")
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            items.append(line)
    return items


def write_results(path: str, rows: List[Tuple[str, List[str], List[str], List[str]]]):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("query\tuni\ttype\tkeyword\n")
        for q, uni, typ, kwd in rows:
            f.write(f"{q}\t{'|'.join(uni)}\t{'|'.join(typ)}\t{'|'.join(kwd)}\n")


# ------------------------------------------------------------
# 메인 실행
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default="test.txt", help="질문 리스트 파일 경로")
    parser.add_argument("--output", "-o", default="results.tsv", help="결과 TSV 경로")
    parser.add_argument("--topn", type=int, default=10, help="키워드 상위 N")
    parser.add_argument("--uni_train", default="KORBERT_NER_UNI/data/train.tsv")
    parser.add_argument("--uni_eval", default="KORBERT_NER_UNI/data/eval_test.tsv")
    parser.add_argument("--type_train", default="KORBERT_NER_TYPE/data/train.tsv")
    parser.add_argument("--type_eval", default="KORBERT_NER_TYPE/data/eval.tsv")
    parser.add_argument("--kw_train", default="KORBERT_NER_KEYWORD/data/train.tsv")
    parser.add_argument("--kw_eval", default="KORBERT_NER_KEYWORD/data/eval.tsv")
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    extractor = AllExtractor(
        topn_keyword=args.topn,
        uni_tsvs=[args.uni_train, args.uni_eval],
        type_tsvs=[args.type_train, args.type_eval],
        kw_tsvs=[args.kw_train, args.kw_eval],
    )

    questions = read_questions(args.input)
    rows = []
    per_line_ms = []

    print_len = len(str(len(questions))) or 1
    print(f"총 {len(questions)}개의 문장 처리 시작...\n")

    for idx, q in enumerate(questions, 1):
        start = time.perf_counter()
        out = extractor.extract(q)
        end = time.perf_counter()
        elapsed_ms = (end - start) * 1000.0
        per_line_ms.append(elapsed_ms)

        uni, typ, kwd = out["uni"], out["type"], out["keyword"]
        print("-" * 78)
        print(f"[{idx:>{print_len}}/{len(questions)}] Q: {q}")
        print("uni    :", uni)
        print("type   :", typ)
        print("keyword:", kwd)
        print(f"⏱ 처리시간: {elapsed_ms:.2f} ms")

        rows.append((q, uni, typ, kwd))

    write_results(args.output, rows)
    print("-" * 78)
    if per_line_ms:
        avg_ms = statistics.mean(per_line_ms)
        print(f"문장 수: {len(per_line_ms)}")
        print(f"평균 처리 시간: {avg_ms:.2f} ms  ({avg_ms/1000:.3f} s)")
    print(f"저장 완료: {os.path.abspath(args.output)}")


# ------------------------------------------------------------
# 실행 시간 측정 (전체)
# ------------------------------------------------------------
if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    total = end_time - start_time
    print(f"\n총 실행 시간: {total:.2f}초")
