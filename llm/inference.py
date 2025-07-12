import openai
import os

# API 키는 환경변수나 직접 입력
openai.api_key = os.getenv("OPENAI_API_KEY")

def ask_gpt(prompt: str, model="gpt-4", temperature=0.2, max_tokens=1024) -> str:
    """
    OpenAI GPT 모델에 프롬프트 전달하고 응답 반환

    Args:
        prompt (str): GPT에게 전달할 프롬프트
        model (str): 사용할 GPT 모델 (기본 gpt-4)
        temperature (float): 창의성 조절
        max_tokens (int): 응답 최대 길이

    Returns:
        str: GPT 응답 텍스트
    """
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "너는 대학교 입시 정보를 요약해주는 AI야. 정확하고 간결하게 응답해."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response["choices"][0]["message"]["content"].strip()

    except Exception as e:
        return f"[ERROR] GPT 호출 실패: {str(e)}"
