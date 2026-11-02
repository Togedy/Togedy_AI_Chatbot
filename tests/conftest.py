import pytest


@pytest.fixture(autouse=True)
def disable_live_llm_in_unit_tests(monkeypatch):
    import generate_answers

    def offline(**kwargs):
        raise RuntimeError("Unit tests do not make paid API calls")

    monkeypatch.setattr(generate_answers, "call_llm", offline)
