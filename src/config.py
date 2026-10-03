"""Загрузка конфигурации проекта: переменные окружения и sources.yaml."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Абсолютные пути проекта
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
EXAMPLES_DIR = ROOT_DIR / "examples"
DB_PATH = DATA_DIR / "articles.db"
SOURCES_PATH = ROOT_DIR / "sources.yaml"
LLM_CACHE_PATH = CACHE_DIR / "llm_cache.json"

DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
EXAMPLES_DIR.mkdir(exist_ok=True)

load_dotenv(ROOT_DIR / ".env")


def _env_list(name: str, default: str) -> list[str]:
    """Читает список из строки, разделённой запятыми."""
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


class Settings:
    """Настройки пайплайна, считываются из переменных окружения."""

    # выбор провайдера
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gigachat")

    # GigaChat
    GIGACHAT_CLIENT_ID: str = os.getenv("GIGACHAT_CLIENT_ID", "")
    GIGACHAT_AUTH_KEY: str = os.getenv("GIGACHAT_AUTH_KEY", "")
    GIGACHAT_SCOPE: str = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
    GIGACHAT_MODEL: str = os.getenv("GIGACHAT_MODEL", "GigaChat")

    # OpenRouter (резервный провайдер, работает только через VPN)
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_MODELS: list[str] = _env_list(
        "OPENROUTER_MODELS",
        "deepseek/deepseek-r1:free,"
        "qwen/qwen3.6-plus:free,"
        "google/gemma-4-26b-a4b-it:free,"
        "meta-llama/llama-3.3-70b-instruct:free,"
        "mistralai/mistral-small-3.2-24b-instruct:free",
    )

    # параметры пайплайна
    TOP_K_ARTICLES: int = int(os.getenv("TOP_K_ARTICLES", "25"))
    N_CLUSTERS: int = int(os.getenv("N_CLUSTERS", "5"))
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
    )
    EMBEDDING_WEIGHT: float = float(os.getenv("EMBEDDING_WEIGHT", "0.7"))
    BM25_WEIGHT: float = float(os.getenv("BM25_WEIGHT", "0.3"))
    ENABLE_LLM_CACHE: bool = os.getenv("ENABLE_LLM_CACHE", "true").lower() == "true"

    DEFAULT_INTERESTS: list[str] = [
        "КИИ",
        "АСУ ТП",
        "уязвимости SCADA",
        "импортозамещение",
        "ИБ в энергетике",
    ]


settings = Settings()


def load_sources() -> list[dict[str, Any]]:
    """Возвращает список включённых источников из sources.yaml."""
    if not SOURCES_PATH.exists():
        return []
    with SOURCES_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    sources = data.get("sources", [])
    return [s for s in sources if s.get("enabled", True)]