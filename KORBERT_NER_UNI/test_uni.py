# test_uni.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from utils import get_label_list
from postprocess import postprocess_ner_output
from lexicon import constrain_tags  # ★ 추가

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
CKPT_DIR = os.path.join(ROOT, "results", "uni_best")
LABEL_PATH = os.path.join(DATA_DIR, "label.txt")

def load_model():
    if not os.path.isdir(CKPT_DIR):
        raise FileNotFoundError(
            f"Checkpoint not found: {CKPT_DIR}\nRun `python trainer_uni.py` first."
        )
    tokenizer = AutoTokenizer.from_pretrained(CKPT_DIR, use_fast=False)
    model = AutoModelForTokenClassification.from_pretrained(CKPT_DIR)
    model.eval()
    return tokenizer, model

def predict_sentence(text: str):
    labels, label_to_id, id_to_label = get_label_list(LABEL_PATH)
    tokenizer, model = load_model()

    words = text.split()

    # 수동 WordPiece 시퀀스 구성 (비-fast 토크나이저 호환)
    cls_id = tokenizer.cls_token_id or tokenizer.convert_tokens_to_ids("[CLS]")
    sep_id = tokenizer.sep_token_id or tokenizer.convert_tokens_to_ids("[SEP]")
    pad_id = tokenizer.pad_token_id or tokenizer.convert_tokens_to_ids("[PAD]")

    wp_ids = [cls_id]
    word_first_wp = []
    for w in words:
        wp = tokenizer.tokenize(w) or ["[UNK]"]
        word_first_wp.append(len(wp_ids))
        wp_ids += tokenizer.convert_tokens_to_ids(wp)
    wp_ids += [sep_id]
    attention = [1] * len(wp_ids)

    # padding
    max_len = 128
    if len(wp_ids) < max_len:
        pad_len = max_len - len(wp_ids)
        wp_ids += [pad_id] * pad_len
        attention += [0] * pad_len
    else:
        wp_ids = wp_ids[:max_len]
        attention = attention[:max_len]

    input_ids = torch.tensor([wp_ids])
    attention_mask = torch.tensor([attention])

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    preds = out.logits[0].argmax(-1).tolist()

    id2lab = {i: l for i, l in enumerate(labels)}
    per_word = [id2lab[preds[idx]] for idx in word_first_wp]

    # ★ 사전(lexicon) 제약 디코딩: 학교 사전에 일치하는 스팬만 B/I로 인정, 나머지는 O
    per_word = constrain_tags(words, per_word)

    result = postprocess_ner_output(words, per_word)
    print("Input: ", text)
    print("Tokens:", words)
    print("Tags:  ", per_word)
    print("UNI:   ", result["UNI"])
    print("\n")

if __name__ == "__main__":
    examples = [
    "건대랑 서연고 수시 모집 일정 알려줘",
    "건국대랑 연세대랑 붙으면 누가 이겨?",
    "서성한에서 입결 누가 더 높아?",
    "연세대 출신 연예인 누구 있어?"
    ] 
    for s in examples:
        predict_sentence(s)
