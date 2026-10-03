"""Сбор публикаций из RSS-источников."""
from __future__ import annotations

import logging
import time
from typing import Any

import feedparser
import requests

from src.config import load_sources
from src.models import Article

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (compatible; DigestAssistant/0.1; +https://example.org)"
)
REQUEST_TIMEOUT = 15
RETRY_DELAY = 2
MAX_RETRIES = 2


def _fetch_feed(url: str) -> bytes | None:
    """Скачивает содержимое RSS-ленты с повторными попытками."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                url,
                timeout=REQUEST_TIMEOUT,
                headers={"User-Agent": USER_AGENT},
            )
            response.raise_for_status()
            return response.content
        except requests.RequestException as e:
            logger.warning(
                "Попытка %d/%d не удалась для %s: %s",
                attempt,
                MAX_RETRIES,
                url,
                e,
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    return None


def _parse_entry(entry: Any, source_name: str, tags: list[str]) -> Article:
    """Преобразует запись RSS в объект Article."""
    title = (entry.get("title") or "").strip()
    link = (entry.get("link") or "").strip()
    summary = (
        entry.get("summary")
        or entry.get("description")
        or ""
    ).strip()
    published = (entry.get("published") or entry.get("updated") or "").strip()

    return Article(
        title=title,
        link=link,
        source=source_name,
        summary=summary,
        published=published,
        tags=list(tags),
    )


def collect_from_source(source: dict[str, Any]) -> list[Article]:
    """Собирает публикации из одного источника."""
    name = source.get("name", "unknown")
    url = source.get("url", "")
    tags = source.get("tags", [])

    if not url:
        logger.warning("Источник %s не содержит url", name)
        return []

    content = _fetch_feed(url)
    if content is None:
        logger.warning("Не удалось получить ленту источника %s", name)
        return []

    feed = feedparser.parse(content)
    if feed.bozo and not feed.entries:
        logger.warning("Лента источника %s повреждена или пуста", name)
        return []

    articles = [_parse_entry(e, name, tags) for e in feed.entries]
    logger.info("Источник %s: получено %d публикаций", name, len(articles))
    return articles


def collect_all() -> list[Article]:
    """Собирает публикации из всех источников, указанных в sources.yaml."""
    sources = load_sources()
    if not sources:
        logger.warning("Нет включённых источников в sources.yaml")
        return []

    all_articles: list[Article] = []
    for source in sources:
        try:
            all_articles.extend(collect_from_source(source))
        except Exception as e:
            logger.exception("Ошибка при сборе из %s: %s", source.get("name"), e)

    logger.info("Всего собрано публикаций: %d", len(all_articles))
    return all_articles