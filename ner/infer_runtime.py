# ner/infer_runtime.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Tuple, Dict, Optional
import os
import logging
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# postprocess import (견고하게)
try:
    from ner.postprocess import postprocess_ner_output
except Exception:
    try:
        from .postprocess import postprocess_ner_output
    except Exception:
        from postprocess import postprocess_ner_output  # 최후의 수단

# label.txt 로더
try:
    from utils import load_label  # utils/__init__.py에 구현
except Exception:
    load_label = None

logger = logging.getLogger("ner.infer_runtime")


def _get_device(explicit: Optional[str] = None) -> torch.device:
    if explicit:
        return torch.device(explicit)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model_and_tokenizer(
    model_dir: str,
    base_model_name: str,
    use_fast: bool = False
):
    """
    tokenizer는 base model에서, model은 학습 체크포인트에서 로드
    - test.py와 동일한 로딩 방식
    """
    if not model_dir or not os.path.isdir(model_dir):
        raise FileNotFoundError(f"NER 모델 디렉터리를 찾을 수 없습니다: {model_dir}")
    if not base_model_name:
        raise RuntimeError("base_model_name이 비어 있습니다 (예: skt/kobert-base-v1).")

    logger.info("Loading tokenizer(base)=%s | model(ckpt)=%s", base_model_name, model_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=use_fast)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    return model, tokenizer


def predict_tags(
    sentence: str,
    model_dir: str,
    base_model_name: str,
    label_path: Optional[str] = None,
    max_length: int = 128,
    device: Optional[str] = None
) -> Tuple[List[str], List[str]]:
    """
    test.py 방식 그대로:
    - 공백으로 나눈 words 사용
    - is_split_into_words=True
    - label.txt의 id→label 매핑 사용 (없으면 model.config.id2label로 폴백)
    - CLS/SEP/PAD 제외하고 '토큰 1개당 단어 1개' 정렬 (간단하지만 test.py와 동일)
    """
    dev = _get_device(device)
    model, tokenizer = _load_model_and_tokenizer(model_dir, base_model_name, use_fast=False)
    model.to(dev)
    model.eval()

    words = sentence.strip().split()
    logger.info("NER predict(test.py-style): device=%s | words=%s", dev, words)

    enc = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length
    )
    enc.pop("token_type_ids", None)
    enc = {k: v.to(dev) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits  # [1, seq_len, num_labels]

    pred_ids = torch.argmax(logits, dim=2)[0].tolist()

    # 라벨 매핑: label.txt 우선, 없으면 model.config.id2label
    if label_path and os.path.exists(label_path) and load_label is not None:
        id2label = load_label(label_path)  # dict[int]->str
    else:
        cfg_map = model.config.id2label
        # huggingface 형식이 str key일 수도 있어요
        id2label = {int(k): v for k, v in cfg_map.items()} if isinstance(cfg_map, dict) else {i: lab for i, lab in enumerate(cfg_map)}

    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])

    aligned_tokens: List[str] = []
    aligned_tags:   List[str] = []

    word_idx = 0
    for idx, tok in enumerate(tokens):
        if tok in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
        # 간단 정렬: subword 여부 무시하고 한 토큰마다 한 단어로 매칭 (test.py 방식)
        if word_idx < len(words):
            aligned_tokens.append(words[word_idx])
            aligned_tags.append(id2label[pred_ids[idx]])
            word_idx += 1

    # 디버그 덤프 (환경변수 NER_DUMP=1이면 자세히)
    if os.getenv("NER_DUMP", "0") == "1":
        for t, l in zip(aligned_tokens, aligned_tags):
            logger.debug("TOK=%s\tTAG=%s", t, l)

    return aligned_tokens, aligned_tags


def run_ner(
    question: str,
    model_dir: Optional[str] = None,
    device: Optional[str] = None
) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    """
    app/main.py에서 호출하는 진입점.
    - .env 예시:
        NER_MODEL_DIR=results/checkpoint-12375
        NER_BASE_MODEL_NAME=skt/kobert-base-v1
        NER_LABEL_PATH=data/label.txt
    """
    model_dir = model_dir or os.getenv("NER_MODEL_DIR")
    base_model = os.getenv("NER_BASE_MODEL_NAME", "skt/kobert-base-v1")
    label_path = os.getenv("NER_LABEL_PATH", "data/label.txt")

    if not model_dir:
        raise RuntimeError("model_dir를 지정하거나 환경변수 NER_MODEL_DIR를 설정하세요.")

    tokens, tags = predict_tags(
        question,
        model_dir=model_dir,
        base_model_name=base_model,
        label_path=label_path,
        max_length=int(os.getenv("NER_MAX_LENGTH", "128")),
        device=device
    )

    ents = postprocess_ner_output(tokens, tags)  # 우리 후처리로 UNI/TYPE/KEYWORD 파싱
    ner_out = {
        "UNI": [str(x) for x in ents.get("UNI", [])],
        "TYPE": [str(x) for x in ents.get("TYPE", [])],
        "KEYWORD": [str(x) for x in ents.get("KEYWORD", [])],
    }
    logger.info("NER out(test.py-style): %s", ner_out)
    return tokens, tags, ner_out
