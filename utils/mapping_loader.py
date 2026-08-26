# utils/mapping_loader.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config_path(env_name: str, default_name: str) -> str:
    value = os.getenv(env_name)
    if value:
        path = Path(value).expanduser()
        return str(path if path.is_absolute() else PROJECT_ROOT / path)
    return str(PROJECT_ROOT / "data" / default_name)


NORMAL_PATH = _config_path("NORMALIZATION_PATH", "normalization_dict.json")
SLUG_PATH = _config_path("SLUG_MAP_PATH", "slug_map.json")

# ===== [1] 기본 내장 매핑 딕셔너리 =====
# 공식 한글 대학명만 포함 (별칭/약칭 제외)
_DEFAULT_UNI_SLUG = {
    "건국대": "konkuk",
    "고려대": "korea",
    "서강대": "sogang",
    "서울대": "seoul",
    "성균관대": "skku",
    "연세대": "yonsei",
    "한양대": "hanyang"
}

@lru_cache(maxsize=None)
def load_normalization(path: str = NORMAL_PATH) -> Dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"UNI": {}, "TYPE": {}, "KEYWORD": {}}

@lru_cache(maxsize=None)
def load_slugmap(path: str = SLUG_PATH) -> Dict:
    # 파일이 있으면 JSON 우선
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    # 없으면 기본값 + 내장 딕셔너리 반환
    return {
        "UNI_SLUG": _DEFAULT_UNI_SLUG,
        "TYPE_SLUG": {"수시": "susi", "정시": "jungsi"}
    }

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
