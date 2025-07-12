import json
import os

# 저장 경로 (Togedy_AI_Chatbot/ner/rule/label_store.json)
STORE_PATH = os.path.join(os.path.dirname(__file__), "label_store.json")

def get_label_store():
    """
    저장된 라벨(UNI, TYPE)을 반환
    """
    if not os.path.exists(STORE_PATH):
        return {"UNI": "", "TYPE": ""}

    try:
        with open(STORE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {"UNI": data.get("UNI", ""), "TYPE": data.get("TYPE", "")}
    except Exception:
        return {"UNI": "", "TYPE": ""}

def update_label_store(label_dict):
    """
    현재 질문의 라벨 중 UNI, TYPE만 저장 (KEYWORD는 제외)

    Args:
        label_dict (dict): {"UNI": ..., "TYPE": ..., "KEYWORD": ...}
    """
    store_data = {
        "UNI": label_dict.get("UNI", ""),
        "TYPE": label_dict.get("TYPE", "")
    }

    with open(STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(store_data, f, ensure_ascii=False, indent=2)
