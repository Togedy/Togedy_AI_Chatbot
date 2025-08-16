# document_retrieval/retriever.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional, Iterable
import os, json, glob, pickle, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.mapping_loader import (
    normalize_uni, normalize_type, uni_to_slug, type_to_slug
)

@dataclass
class DocItem:
    doc_id: str
    uni: str
    dtype: str
    source: str     # 파일 경로
    text: str
    uni_key: str = ""   # ex) "seoul", "hanyang"
    type_key: str = ""  # ex) "susi", "jungsi"

def _type_key_from_filename(fname: str) -> str:
    f = fname.lower()
    if f.startswith("susi_") or "susi_" in f: return "susi"
    if f.startswith("jungsi_") or "jungsi_" in f: return "jungsi"
    return ""

class TfidfRetriever:
    def __init__(self, ngram_range=(1, 2), min_df=1, max_df=0.95):
        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df, max_df=max_df)
        self.doc_mat = None
        self.docs: List[DocItem] = []

    # ------------------ TXT 우선 코퍼스 ------------------
    def load_txt_corpus(self,
                        uni_keys: Optional[Iterable[str]] = None,
                        type_keys: Optional[Iterable[str]] = None) -> List[DocItem]:
        items: List[DocItem] = []
        root = os.getcwd()
        uni_root = os.path.join(root, "university")
        if not os.path.isdir(uni_root):
            return items

        uni_keys = set(uni_keys or [])
        type_keys = set(type_keys or [])

        target_unis = list(uni_keys) if uni_keys else [
            d for d in os.listdir(uni_root) if os.path.isdir(os.path.join(uni_root, d))
        ]
        for uk in target_unis:
            udir = os.path.join(uni_root, uk)
            for path in glob.glob(os.path.join(udir, "*_text.txt")):
                tk = _type_key_from_filename(os.path.basename(path))
                if type_keys and tk not in type_keys:
                    continue
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                    if not text:
                        continue
                    items.append(DocItem(
                        doc_id=f"{uk}_{tk}_{os.path.basename(path)}",
                        uni="", dtype="", source=path, text=text[:12000],
                        uni_key=uk, type_key=tk
                    ))
                except Exception:
                    continue
        return items

    # ------------------ 혼합 코퍼스(폴백) ------------------
    def load_corpus(self,
                    uni_keys: Optional[Iterable[str]] = None,
                    type_keys: Optional[Iterable[str]] = None) -> List[DocItem]:
        items: List[DocItem] = []
        root = os.getcwd()
        uni_keys = set(uni_keys or [])
        type_keys = set(type_keys or [])

        json_path = os.path.join(root, "data", "document_chunks.json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for i, d in enumerate(data):
                uk = d.get("uni_key") or uni_to_slug(normalize_uni(d.get("uni", "")))
                tk = d.get("type_key") or type_to_slug(normalize_type(d.get("type", "")))
                if uni_keys and uk not in uni_keys: continue
                if type_keys and tk not in type_keys: continue
                items.append(DocItem(
                    doc_id=d.get("id", f"json_{i}"),
                    uni=d.get("uni",""), dtype=d.get("type",""),
                    source=d.get("source","data/document_chunks.json"),
                    text=d.get("text",""),
                    uni_key=uk, type_key=tk
                ))
            if items:
                return items

        uni_root = os.path.join(root, "university")
        if os.path.isdir(uni_root):
            target_unis = list(uni_keys) if uni_keys else [
                d for d in os.listdir(uni_root) if os.path.isdir(os.path.join(uni_root, d))
            ]
            for uk in target_unis:
                udir = os.path.join(uni_root, uk)
                for pat in ["*_tables_cleaned.csv", "*_tables.csv", "*_text.txt"]:
                    for path in glob.glob(os.path.join(udir, pat)):
                        tk = _type_key_from_filename(os.path.basename(path))
                        if type_keys and tk not in type_keys:
                            continue
                        try:
                            if path.lower().endswith(".csv"):
                                import csv
                                with open(path, "r", encoding="utf-8-sig") as f:
                                    reader = csv.reader(f)
                                    for rix, row in enumerate(reader):
                                        text = " ".join(c for c in row if c)[:4000]
                                        if not text.strip(): continue
                                        items.append(DocItem(
                                            doc_id=f"{uk}_{tk}_{os.path.basename(path)}_{rix}",
                                            uni="", dtype="", source=path, text=text,
                                            uni_key=uk, type_key=tk
                                        ))
                            else:
                                with open(path, "r", encoding="utf-8") as f:
                                    text = f.read().strip()
                                if text:
                                    items.append(DocItem(
                                        doc_id=f"{uk}_{tk}_{os.path.basename(path)}",
                                        uni="", dtype="", source=path, text=text[:12000],
                                        uni_key=uk, type_key=tk
                                    ))
                        except Exception:
                            continue
        return items

    # ------------------ 인덱싱(작은 코퍼스 안전 보정) ------------------
    def build(self, docs: List[DocItem]):
        self.docs = docs
        n_docs = len(docs)
        if n_docs <= 3:
            self.vectorizer.set_params(min_df=1, max_df=1.0)
        try:
            texts = [(d.text or "") for d in docs]
            self.doc_mat = self.vectorizer.fit_transform(texts)
        except ValueError:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 1), min_df=1, max_df=1.0)
            self.doc_mat = self.vectorizer.fit_transform(texts)

    def save(self, index_path="data/tfidf_index.pkl", meta_path="data/tfidf_meta.json"):
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        with open(index_path, "wb") as f:
            pickle.dump({"vec": self.vectorizer, "mat": self.doc_mat}, f)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump([d.__dict__ for d in self.docs], f, ensure_ascii=False, indent=2)

    def load(self, index_path="data/tfidf_index.pkl", meta_path="data/tfidf_meta.json"):
        with open(index_path, "rb") as f:
            obj = pickle.load(f)
        self.vectorizer = obj["vec"]
        self.doc_mat = obj["mat"]
        with open(meta_path, "r", encoding="utf-8") as f:
            self.docs = [DocItem(**x) for x in json.load(f)]

    # ------------------ 쿼리/검색 ------------------
    @staticmethod
    def _compose_query(validated: Dict[str, Any]) -> str:
        ents = validated.get("entities", {})
        def grab(key):
            arr = ents.get(key, [])
            return [x.get("normalized") or x.get("text")
                    for x in arr if (x.get("normalized") or x.get("text"))]
        toks = []
        toks += grab("UNI"); toks += grab("TYPE"); toks += grab("KEYWORD")
        seen, out = set(), []
        for t in toks:
            if t and (t not in seen):
                out.append(t); seen.add(t)
        return " ".join(out)

    @staticmethod
    def extract_filters(validated: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        ents = validated.get("entities", {})
        uni_keys, type_keys = set(), set()
        for x in ents.get("UNI", []):
            can = normalize_uni(x.get("normalized") or x.get("text") or "")
            slug = uni_to_slug(can)
            if slug: uni_keys.add(slug)
        for x in ents.get("TYPE", []):
            can = normalize_type(x.get("normalized") or x.get("text") or "")
            slug = type_to_slug(can)
            if slug: type_keys.add(slug)
        return list(uni_keys), list(type_keys)

    def search(self,
               validated_json: Dict[str, Any],
               top_k: int = 3,
               min_score: float = 0.0) -> List[Tuple[DocItem, float]]:
        if self.doc_mat is None or not self.docs:
            return []
        q = self.vectorizer.transform([self._compose_query(validated_json).strip() or ""])
        sims = cosine_similarity(q, self.doc_mat).ravel()
        idxs = sims.argsort()[::-1][:max(1, top_k)]
        hits = [(self.docs[i], float(sims[i])) for i in idxs]
        if min_score > 0.0:
            hits = [h for h in hits if h[1] >= min_score]
        return hits

    # ------------------ 페이지 랭킹 ------------------
    def rank_pages(self,
                   text_path: str,
                   query: str,
                   top_n: int = 3,
                   page_regex: str = r"^==== Page (\d+) ====",
                   min_chars: int = 200) -> List[Dict[str, Any]]:
        if not os.path.exists(text_path):
            return []
        with open(text_path, "r", encoding="utf-8") as f:
            full = f.read()

        lines = full.splitlines()
        pages, cur_page_no, cur_buf = [], None, []
        page_pat = re.compile(page_regex)
        for ln in lines:
            m = page_pat.match(ln.strip())
            if m:
                if cur_page_no is not None and cur_buf:
                    pages.append((cur_page_no, "\n".join(cur_buf).strip()))
                cur_page_no = int(m.group(1))
                cur_buf = []
            else:
                cur_buf.append(ln)
        if cur_page_no is not None and cur_buf:
            pages.append((cur_page_no, "\n".join(cur_buf).strip()))
        if not pages:
            pages = [(1, full)]

        cleaned = [(p, t) for (p, t) in pages if t and len(t) >= min_chars]
        if not cleaned:
            cleaned = pages

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vec = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_df=1.0)
        mat = vec.fit_transform([t for _, t in cleaned])
        q = vec.transform([query or ""])
        sims = cosine_similarity(q, mat).ravel()

        idxs = sims.argsort()[::-1][:max(1, top_n)]
        out = []
        for i in idxs:
            page_no, text = cleaned[i]
            score = float(sims[i])
            excerpt = text[:400].replace("\n", " ")
            out.append({"page": page_no, "score": score, "excerpt": excerpt})
        return out

    # ------------------ TXT 결과의 이웃 리소스 묶기 ------------------
    @staticmethod
    def bundle_neighbors(text_path: str) -> Dict[str, Optional[str]]:
        d = os.path.dirname(text_path)
        base = os.path.basename(text_path).lower()
        stem = "susi" if "susi_" in base else ("jungsi" if "jungsi_" in base else "")
        if not stem:
            return {"text": text_path, "tables": None, "pdf": None}

        def find_one(patterns):
            for pat in patterns:
                hits = glob.glob(os.path.join(d, pat))
                if hits:
                    return hits[0]
            return None

        tables = find_one([f"{stem}_tables_cleaned.csv", f"{stem}_tables.csv"])
        pdf = find_one([f"{stem}.pdf", f"{stem}_*.pdf"])
        txt = os.path.join(d, f"{stem}_text.txt")
        if not os.path.exists(txt):
            txt = text_path

        return {"text": txt, "tables": tables, "pdf": pdf}
