from settings import PROJECT_ROOT, Settings
import pytest
from settings import llm_timeout_seconds


def test_project_root_contains_runtime_data():
    assert (PROJECT_ROOT / "main.py").is_file()
    assert (PROJECT_ROOT / "university").is_dir()


def test_settings_defaults(monkeypatch):
    for name in ("OPENAI_MODEL", "APP_HOST", "APP_PORT", "APP_DEBUG"):
        monkeypatch.delenv(name, raising=False)

    config = Settings.from_env()

    assert config.openai_model == "gpt-4o-mini"
    assert config.host == "0.0.0.0"
    assert config.port == 5000
    assert config.debug is False


@pytest.mark.parametrize("value", ["0", "-1", "inf", "nan", "301", "bad"])
def test_invalid_llm_timeout_is_rejected(monkeypatch, value):
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", value)
    with pytest.raises(ValueError):
        llm_timeout_seconds()


def test_llm_timeout_can_be_configured(monkeypatch):
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "12.5")
    assert llm_timeout_seconds() == 12.5
