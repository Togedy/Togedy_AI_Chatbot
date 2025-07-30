# llm/inference.py

import os
import openai

# OpenAI API 키는 환경변수에서 불러오도록 설정
openai.api_key = os.getenv("OPENAI_API_KEY")


def generate_answer(prompt: str, model: str = "gpt-4") -> str:
    """
    OpenAI GPT API를 사용해 답변을 생성합니다.
    :param prompt: GPT에 전달할 전체 프롬프트 문자열
    :param model: 사용할 모델 이름 (기본: gpt-4)
    :return: 생성된 답변 문자열
    """
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "당신은 대학 입시에 대해 정확하고 친절하게 설명하는 AI입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )

        return response.choices[0].message["content"].strip()

    except Exception as e:
        return f"[GPT 호출 오류] {str(e)}"
