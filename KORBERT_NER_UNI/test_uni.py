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
    "건대 수시 모집 일정 알려줘",
    "연세대 정시 전형 방법과 제출서류",
    "영어등급 환산표와 KU 일반학생 전형 안내",
    "건대랑 서연고 수시 모집 일정 알려줘",
    "나 서울대 정시랑 수시에서 컴퓨터공학과 몇명 뽑는지 궁금해",
    "부산 사는 사람인데 건대가 메리트 있을까?",
    "연세대랑 고려대중에 어디가 더 좋아?",
    "연고대중에서 학교를 어디를 써야 좋을지 고민중이야",
    "경희대 소웨과랑 건대 컴공과 둘다 붙으면 어디로 가는게 좋을까?",
    "건대 컴공과 논술 최저가 있는 거야? 없는거야?",
    "작년 서성한 중경외시 중에 인원 미달인 학교를 알려줘",
    "실기 없는 미대 입시 전형 정리해줘",
    "건대 수시랑 연대 수시 중에 서양회과 예비 번호는 어디가 더 잘빠져?",
    "건국대 학생부 종합전형을 쓰려고 하는데 컴퓨터공학부말고 또 경쟁률이 6:1이랑 비슷한 곳이 있나요?",
    "내가 공부를 못해서 실기 전형 60%이상인 미대 전형을 찾아줘",
    "내가 실기를 못해서 수능 전형 60%이상인 미대 전형을 찾아줘",
    "건국대 자유전공공학부 들어가는 방법에 대해 알려줘",
    "타학교 산업공학과 목표로 물리를 안들었는데 건국대 학생부종합 지원하려면 불이익이 클까요? 대부분 공대는 물리 안들으면 학생부 종합 전형 광탈이라고 하니까요... 내신도 3.0이예요 일반고 생기부 열심히 챙기긴 했는데 힘들까요? 자유전공학부는 어떤가요?",
    "건대 경영 가려면 정시 백분위 몇 정도 맞아야 되는지 궁금합니다",
    "건국대 경영관 위치나 시설 좋은가요?",
    "현대미술과 희망하는데 학생부 종합전형에서 기타과목 중요도가 궁금합니다",
    "나 심심해"
    ] 
    for s in examples:
        predict_sentence(s)
