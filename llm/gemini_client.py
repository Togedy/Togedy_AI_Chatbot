# -*- coding: utf-8 -*-
import os
from typing import Any, Dict
import google.generativeai as genai

class GeminiClient:
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise RuntimeError("GOOGLE_API_KEY가 설정되어 있지 않습니다.")
        genai.configure(api_key=self.api_key)
        self.model_name = model or os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
        self._model = genai.GenerativeModel(self.model_name)

    def generate_json(self, system_instruction: str, user_prompt: str) -> Dict[str, Any]:
        """
        Gemini에게 JSON만 받도록 강제.
        """
        resp = self._model.generate_content(
            contents=[
                {"role": "user", "parts": [system_instruction]},
                {"role": "user", "parts": [user_prompt]},
            ],
            generation_config={
                "temperature": 0.0,
                "response_mime_type": "application/json",
            },
        )
        # safety/blocked 대응
        if not hasattr(resp, "text") or not resp.text:
            raise RuntimeError("Gemini 응답이 비어있습니다.")
        import json
        return json.loads(resp.text)
