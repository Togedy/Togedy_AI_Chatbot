import re
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from utils import load_label_list

MODEL_DIR = "./KORBERT_NER_TYPE/results/best_model"

JOOSA_SUFFIXES = [
    "와", "과", "랑", "하고", "및",
    "은", "는", "이", "가", "을", "를", "만", "도",
    "에", "에서", "으로", "로", "께", "께서", "께서는",
    "보다", "부터", "까지", "마저", "조차", "마다",
    "이나", "나", "이나마", "나마",
    "든지", "라도", "라면", "라도",
    "와의", "과의", "의",
]

ALLOWED_TYPES = ("수시", "정시")

def split_korean_tokens(sentence: str):
    raw = sentence.strip().split()
    out = []
    for tok in raw:
        if tok.startswith("수시") or tok.startswith("정시"):
            core = "수시" if tok.startswith("수시") else "정시"
            rest = tok[len(core):]
            if rest:
                matched = False
                for suf in sorted(JOOSA_SUFFIXES, key=len, reverse=True):
                    if rest == suf:
                        out.extend([core, suf])
                        matched = True
                        break
                if not matched:
                    if re.fullmatch(r"[가-힣]+", rest):
                        out.extend([core, rest])
                    else:
                        out.append(tok)
            else:
                out.append(core)
        else:
            out.append(tok)
    return out

def _normalize_type_token(tok: str):
    """
    TYPE는 수시/정시만 허용.
    - 토큰이 정확히 수시/정시면 그대로
    - 혹시 '수시...'/'정시...' 형태면 앞부분만 정규화
    """
    if tok == "수시" or tok.startswith("수시"):
        return "수시"
    if tok == "정시" or tok.startswith("정시"):
        return "정시"
    return None

def _unique_keep_order(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def predict(sentence: str):
    labels, label2id, id2label = load_label_list()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
    model.eval()

    tokens = split_korean_tokens(sentence)

    enc = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )
    if "token_type_ids" in enc:
        del enc["token_type_ids"]

    with torch.no_grad():
        logits = model(**enc).logits
    pred_ids = logits.argmax(-1)[0].tolist()

    word_ids = enc.word_ids(batch_index=0)
    aligned = []
    used = set()
    for idx, wid in enumerate(word_ids):
        if wid is None or wid in used:
            continue
        used.add(wid)
        aligned.append((tokens[wid], id2label[pred_ids[idx]]))

    # 핵심 수정: 모델이 B/I-TYPE로 찍어도, 결과(TYPE)는 수시/정시만 남김
    types_raw = []
    for tok, tag in aligned:
        if tag.startswith("B-") or tag.startswith("I-"):
            norm = _normalize_type_token(tok)
            if norm in ALLOWED_TYPES:
                types_raw.append(norm)

    types = _unique_keep_order(types_raw)

    return {
        "tokens": [t for t, _ in aligned],
        "tags":   [g for _, g in aligned],
        "TYPE":   types
    }

if __name__ == "__main__":
    print(predict("수시와 정시 둘 다 일정 알려줘"))
