from ner.llm_validate_gemini import validate_with_gemini
import json

if __name__ == "__main__":
    q = "연세대 수시 학생부종합전형에서 소프트웨어학부 전형 방법 알려줘."
    ner_out = {
        "UNI": ["연세대"],
        "TYPE": ["수시", "학생부종합전형"],
        "KEYWORD": ["전형", "방법"]
    }
    res = validate_with_gemini(q, ner_out)  # LLM 검증만 수행
    print(json.dumps(res, ensure_ascii=False, indent=2))
