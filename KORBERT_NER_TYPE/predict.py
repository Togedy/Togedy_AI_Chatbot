import re
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from utils import load_label_list

MODEL_DIR = "./KORBERT_NER_TYPE/results/best_model"

# 수시/정시 뒤에 자주 붙는 조사·어미(필요시 추가)
JOOSA_SUFFIXES = [
    "와", "과", "랑", "하고", "및",
    "은", "는", "이", "가", "을", "를", "만", "도",
    "에", "에서", "으로", "로", "께", "께서", "께서는",
    "보다", "부터", "까지", "마저", "조차", "마다",
    "이나", "나", "이나마", "나마",
    "든지", "라도", "라면", "라도",
    "와의", "과의", "의",
]

# '수시/정시'에 붙은 조사/어미를 분리
def split_korean_tokens(sentence: str):
    raw = sentence.strip().split()
    out = []
    for tok in raw:
        # 수시/정시로 시작하면서 뒤에 한글만 붙은 경우 분해 시도
        if tok.startswith("수시") or tok.startswith("정시"):
            core = "수시" if tok.startswith("수시") else "정시"
            rest = tok[len(core):]
            if rest:  # 붙은 꼬리가 있음
                # 길이가 1~3인 꼬리부터 시도 (와/은/는/이/가/을/를/만/도/에/로/과/의 등)
                matched = False
                # 긴 꼬리부터 우선 매칭 (예: '에서', '까지')
                for suf in sorted(JOOSA_SUFFIXES, key=len, reverse=True):
                    if rest == suf:
                        out.extend([core, suf])
                        matched = True
                        break
                if not matched:
                    # 알 수 없는 꼬리: 가능한 경우만 core 분리, 나머지는 그대로
                    # 예: '수시만요' → ['수시','만요']처럼 단순 분할
                    if re.fullmatch(r"[가-힣]+", rest):
                        out.extend([core, rest])
                    else:
                        out.append(tok)
            else:
                out.append(core)
        else:
            out.append(tok)
    return out

def predict(sentence: str):
    labels, label2id, id2label = load_label_list()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
    model.eval()

    # ✅ 조사/어미 분리 전처리
    tokens = split_korean_tokens(sentence)

    enc = tokenizer(tokens, is_split_into_words=True, return_tensors="pt",
                    truncation=True, max_length=128)
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

    types = [tok for tok, tag in aligned if tag.startswith("B-") or tag.startswith("I-")]
    return {"tokens": [t for t, _ in aligned], "tags": [g for _, g in aligned], "TYPE": types}

if __name__ == "__main__":
    print(predict("수시와 정시 둘 다 일정 알려줘"))
