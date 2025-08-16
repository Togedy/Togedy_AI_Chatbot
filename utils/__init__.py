# utils/__init__.py
# -*- coding: utf-8 -*-
from __future__ import annotations

def load_label(path: str):
    """
    data/label.txt에서 줄 단위로 라벨을 읽어
    id(index) -> label 문자열 매핑을 반환합니다.
    예)
      0 O
      1 B-UNI
      2 I-UNI
      ...
    """
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # "0 O" 또는 "O" 형태 모두 지원
            parts = s.split()
            if len(parts) == 1:
                labels.append(parts[0])
            else:
                labels.append(parts[-1])
    return {i: lab for i, lab in enumerate(labels)}
