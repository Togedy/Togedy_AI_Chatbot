from types import SimpleNamespace
import sys
import types

import generate_answers as answers
from generate_answers import call_llm
from extract_all import gemini_sort


def test_openai_request_uses_configured_timeout_without_sdk_retries(monkeypatch):
    options = {}
    def client(**kwargs):
        options.update(kwargs)
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **k: response)))
    monkeypatch.setattr(answers, "OpenAI", client)
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-only")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "12")
    assert call_llm("hello") == "ok"
    assert options["timeout"] == 12 and options["max_retries"] == 0


def test_gemini_sort_uses_same_request_limit(monkeypatch):
    seen = {}
    def generate(prompt, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(text='{"UNI":"서울대","TYPE":"수시","KEYWORD":["모집인원"]}')
    google = types.ModuleType("google")
    genai = types.ModuleType("google.generativeai")
    genai.configure = lambda **k: None
    genai.GenerativeModel = lambda model: SimpleNamespace(generate_content=generate)
    google.generativeai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.generativeai", genai)
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "15")
    result = gemini_sort("unit-test-only", "model", ["서울대"], ["수시"], ["모집인원"])
    assert result[0]["UNI"] == "서울대"
    assert seen["request_options"] == {"timeout": 15, "retry": None}
