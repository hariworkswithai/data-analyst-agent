"""Application configuration loaded from environment variables and defaults."""

import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
CHART_DIR = BASE_DIR / "generated" / "charts"
REPORT_DIR = BASE_DIR / "generated" / "reports"

for _dir in (UPLOAD_DIR, CHART_DIR, REPORT_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# Load env before the Config instance is created so values are applied.
load_dotenv(BASE_DIR / ".env", override=False)
load_dotenv(BASE_DIR / "backend" / ".env", override=False)


class Config:
    # OpenRouter
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL: str = os.getenv(
        "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
    )
    OPENROUTER_MODEL: str = os.getenv(
        "OPENROUTER_MODEL", "openai/gpt-4o-mini"
    )
    LLM_TIMEOUT_SECONDS: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "90"))

    # Upload limits
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    MAX_ROWS: int = int(os.getenv("MAX_ROWS", "200000"))
    MAX_COLUMNS: int = int(os.getenv("MAX_COLUMNS", "200"))
    ALLOWED_EXTENSIONS: tuple[str, ...] = (".csv",)

    # Workflow limits
    MAX_AGENT_STEPS: int = int(os.getenv("MAX_AGENT_STEPS", "20"))
    MAX_REVIEW_CYCLES: int = int(os.getenv("MAX_REVIEW_CYCLES", "3"))
    MAX_TOOL_RETRIES: int = int(os.getenv("MAX_TOOL_RETRIES", "2"))
    ANALYSIS_TIMEOUT_SECONDS: int = int(
        os.getenv("ANALYSIS_TIMEOUT_SECONDS", "600")
    )

    # LLM behavior
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))

    @property
    def openrouter_configured(self) -> bool:
        return bool(self.OPENROUTER_API_KEY) and "your_key" not in self.OPENROUTER_API_KEY


config = Config()