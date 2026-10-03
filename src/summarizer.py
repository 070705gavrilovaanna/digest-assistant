"""Суммаризация публикаций: extractive fallback и LLM."""
from __future__ import annotations

import logging

from src.llm_client import AllModelsExhaustedError, get_llm
from src.models import Article, Cluster

logger = logging.getLogger(__name__)

MAX_INPUT_CHARS = 1800
FALLBACK_LEN = 300


def _fallback_summary(article: Article) -> str:
    text = article.summary or article.title
    text = text.strip()
    if len(text) <= FALLBACK_LEN:
        return text
    cut = text[:FALLBACK_LEN]
    last_dot = cut.rfind(".")
    if last_dot > 100:
        cut = cut[: last_dot + 1]
    return cut + " ..."


def summarize_article(article: Article) -> str:
    """Суммаризирует одну публикацию через LLM с fallback."""
    raw = article.summary or ""
    if len(raw) < 120:
        return raw or article.title

    prompt = (
        "Сделай краткое резюме публикации на русском языке (2-3 предложения). "
        "Не добавляй ничего от себя, используй только текст ниже.\n\n"
        f"Заголовок: {article.title}\n\n"
        f"Текст: {raw[:MAX_INPUT_CHARS]}"
    )

    try:
        llm = get_llm()
        summary = llm.complete(
            prompt,
            system="Ты редактор ИТ/ИБ-дайджеста. Пиши по-русски, кратко и по делу.",
            temperature=0.2,
            max_tokens=180,
        )
        summary = summary.strip()
        return summary or _fallback_summary(article)
    except AllModelsExhaustedError:
        logger.warning("LLM недоступен, использую fallback для %s", article.title)
    except Exception as e:
        logger.warning("Ошибка LLM при суммаризации: %s", e)

    return _fallback_summary(article)


def summarize_cluster(cluster: Cluster) -> None:
    """Суммаризирует все статьи кластера, обновляя поле llm_summary."""
    for article in cluster.articles:
        article.llm_summary = summarize_article(article)


def summarize_all(clusters: list[Cluster]) -> None:
    """Суммаризирует все кластеры."""
    for cluster in clusters:
        summarize_cluster(cluster)
    logger.info("Суммаризация завершена для %d кластеров", len(clusters))