"""Freeze complete API wording, separately from behavioral flow assertions."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import generate_answers as answers
import main
from evaluations.verify_dialogue_wording import run_scenarios


def test_complete_api_wording_matches_reviewable_capture(monkeypatch):
    expected = json.loads((Path(__file__).parent / "fixtures/dialogue_wording.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(answers, "_extract_ner_direct", lambda *a: {"UNI": [], "TYPE": [], "KEYWORD": []})
    monkeypatch.setattr(answers, "call_llm", lambda **k: pytest.fail("Unexpected external model call"))
    monkeypatch.setattr(main, "get_pipeline", lambda: SimpleNamespace(
        uni_ex=None, type_ex=None, kw_ex=None, api_key="", gemini_model="", openai_model=""))
    actual = run_scenarios(main.create_app().test_client())
    assert len(actual) == len(expected) == 6
    for scenario, captured in zip(actual, expected):
        assert scenario["passed"], scenario["scenario"]
        assert scenario["scenario"] == captured["scenario"]
        responses = [t["response"] for t in scenario["turns"]]
        assert len(responses) == len(captured["responses"])
        for response, snapshot in zip(responses, captured["responses"]):
            # File separators vary across operating systems, wording does not.
            for item in (response, snapshot):
                if "location" in item:
                    item["location"] = item["location"].replace("\\", "/")
            assert response == snapshot, scenario["scenario"]


def test_form_help_is_removed_before_models_can_extract_it():
    seen = []
    def extract(text):
        seen.append(text)
        return []
    uni = SimpleNamespace(extract_uni=extract)
    typ = SimpleNamespace(extract_type=extract)
    kw = SimpleNamespace(extract_keywords=extract)
    result = answers._extract_ner_direct(
        "대학: 연세대\n전형(수시·정시, 알면 세부 전형명도): \n"
        "궁금한 내용(예: 제출서류·일정·모집인원): 제출서류", uni, typ, kw)
    assert len(seen) == 3
    assert all("수시" not in text and "정시" not in text and "일정" not in text for text in seen)
    assert all("제출서류" in text for text in seen)
    assert result["TYPE"] == []
