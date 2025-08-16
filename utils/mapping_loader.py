# utils/mapping_loader.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import json
import os
from functools import lru_cache
from typing import Dict

NORMAL_PATH = os.getenv("NORMALIZATION_PATH", "data/normalization_dict.json")
SLUG_PATH   = os.getenv("SLUG_MAP_PATH",       "data/slug_map.json")

@lru_cache(maxsize=None)
def load_normalization(path: str = NORMAL_PATH) -> Dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"UNI": {}, "TYPE": {}, "KEYWORD": {}}

@lru_cache(maxsize=None)
def load_slugmap(path: str = SLUG_PATH) -> Dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"UNI_SLUG": {}, "TYPE_SLUG": {"수시": "susi", "정시": "jungsi"}}

def normalize_uni(name: str) -> str:
    n = (name or "").strip()
    if not n:
        return ""
    uni_norm = load_normalization().get("UNI", {})
    return uni_norm.get(n, n)

def normalize_type(t: str) -> str:
    s = (t or "").strip()
    if not s:
        return ""
    type_norm = load_normalization().get("TYPE", {})
    if s in type_norm:
        return type_norm[s]
    if "정시" in s: return "정시"
    if "수시" in s: return "수시"
    if any(k in s for k in ["논술", "학생부", "교과", "종합", "특기자"]):
        return "수시"
    return ""

def uni_to_slug(canonical_uni: str) -> str:
    return load_slugmap().get("UNI_SLUG", {}).get(canonical_uni, "")

def type_to_slug(canonical_type: str) -> str:
    return load_slugmap().get("TYPE_SLUG", {}).get(canonical_type, "")
