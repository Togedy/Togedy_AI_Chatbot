# document_retrieval/table_pick.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import pandas as pd
from typing import List, Dict, Any

def pick_rows_by_keywords(csv_path: str,
                          must_keywords: List[str],
                          any_keywords: List[str] = None,
                          max_rows: int = 5) -> List[Dict[str, Any]]:
    any_keywords = any_keywords or []
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    merged = df.astype(str).apply(lambda r: " ".join([x for x in r.values if x and x != "nan"]), axis=1)
    mask = merged.apply(lambda s: all(k in s for k in must_keywords)) if must_keywords else (merged != "")
    if any_keywords:
        mask = mask & merged.apply(lambda s: any(k in s for k in any_keywords))
    sub = df[mask].head(max_rows)
    return sub.to_dict(orient="records")
