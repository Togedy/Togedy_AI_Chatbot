"""Application configuration and project paths.

Keep filesystem and environment lookups here so command-line tools and the
Flask server behave the same regardless of the current working directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
UNIVERSITY_DIR = PROJECT_ROOT / "university"
DATA_DIR = PROJECT_ROOT / "data"


def load_dotenv_file() -> None:
    """Load the project .env when python-dotenv is installed."""
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass


@dataclass(frozen=True)
class Settings:
    google_api_key: str
    openai_api_key: str
    gemini_model: str
    openai_model: str
    host: str
    port: int
    debug: bool

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv_file()
        return cls(
            google_api_key=os.getenv("GOOGLE_API_KEY", ""),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-1.5-pro"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            host=os.getenv("APP_HOST", "0.0.0.0"),
            port=int(os.getenv("APP_PORT", "5000")),
            debug=os.getenv("APP_DEBUG", "false").lower() in {"1", "true", "yes"},
        )
