from settings import PROJECT_ROOT, Settings


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
