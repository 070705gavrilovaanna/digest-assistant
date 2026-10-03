"""Тематическая кластеризация публикаций и генерация названий тем."""
from __future__ import annotations

import logging
import re

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.config import settings
from src.embedder import embed_texts
from src.llm_client import AllModelsExhaustedError, get_llm
from src.models import Article, Cluster

logger = logging.getLogger(__name__)

# Требования к названию темы, чтобы не пустить в дайджест мусор
MIN_TITLE_WORDS = 2
MAX_TITLE_WORDS = 8
MAX_TITLE_CHARS = 80

# Слова-маркеры, которых в названии темы быть не должно
FORBIDDEN_PATTERNS = [
    r"\bапи\b",       # «АЗИЙСКОЙ АПИ» — очевидный сбой
    r"\bapi\b",
    r"^без темы",
    r"^нет данных",
]


def _choose_n_clusters(embeddings: np.ndarray, max_k: int) -> int:
    n_samples = embeddings.shape[0]
    if n_samples < 4:
        return 1

    max_k = min(max_k, n_samples - 1)
    if max_k < 2:
        return 1

    best_k, best_score = 2, -1.0
    for k in range(2, max_k + 1):
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init="auto")
            labels = km.fit_predict(embeddings)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(embeddings, labels)
            if score > best_score:
                best_k, best_score = k, score
        except Exception:
            continue
    return best_k


def _is_valid_title(title: str) -> bool:
    """Проверяет, что название темы выглядит адекватно."""
    if not title:
        return False
    t = title.strip()
    if len(t) > MAX_TITLE_CHARS:
        return False

    words = re.findall(r"\w+", t, flags=re.UNICODE)
    if len(words) < MIN_TITLE_WORDS or len(words) > MAX_TITLE_WORDS:
        return False

    # Если больше 60% слов заглавные — это мусор, а не название
    upper_words = [w for w in words if w.isupper() and len(w) > 2]
    if upper_words and len(upper_words) / len(words) > 0.6:
        return False

    low = t.lower()
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, low):
            return False

    return True


def _fallback_title(articles: list[Article]) -> str:
    """Fallback-название темы: самое частое содержательное слово заголовков."""
    stopwords = {
        "и", "в", "на", "с", "по", "для", "из", "к", "о", "об", "от",
        "the", "a", "an", "of", "to", "in", "for", "on", "with", "and",
        "уязвимости", "уязвимость", "публикация", "cisa", "adds",
        "known", "exploited", "vulnerabilities", "catalog",
    }
    word_freq: dict[str, int] = {}
    for a in articles:
        for word in re.findall(r"[А-Яа-яA-Za-z][А-Яа-яA-Za-z\-]{3,}", a.title):
            w = word.lower()
            if w in stopwords:
                continue
            word_freq[w] = word_freq.get(w, 0) + 1

    if not word_freq:
        return "Без темы"

    top = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:2]
    words = [w.capitalize() for w, _ in top]
    return " и ".join(words)


def _generate_cluster_title(articles: list[Article]) -> str:
    """Генерирует название темы через LLM с валидацией."""
    titles = [a.title for a in articles if a.title][:8]
    if not titles:
        return "Без темы"

    prompt = (
        "Ниже список заголовков публикаций по ИТ и информационной безопасности. "
        "Придумай одно короткое название темы (3-6 слов), объединяющее их. "
        "Не используй аббревиатуры без расшифровки, не пиши всё заглавными "
        "буквами. Ответь только названием, без пояснений и без кавычек.\n\n"
        + "\n".join(f"- {t}" for t in titles)
    )

    for attempt in range(2):
        try:
            llm = get_llm()
            title = llm.complete(
                prompt,
                system=(
                    "Ты редактор технического дайджеста. Отвечай кратко "
                    "на русском языке. Название темы должно быть понятным "
                    "без контекста."
                ),
                temperature=0.2 + attempt * 0.2,
                max_tokens=40,
            )
            title = title.strip().strip('"').strip("'").split("\n")[0].strip()
            if _is_valid_title(title):
                return title
            logger.warning(
                "Название темы не прошло валидацию: %r", title
            )
        except AllModelsExhaustedError:
            logger.warning("LLM недоступен для генерации названия темы")
            break
        except Exception as e:
            logger.warning("Ошибка LLM при генерации названия темы: %s", e)

    fallback = _fallback_title(articles)
    logger.info("Использую fallback-название темы: %s", fallback)
    return fallback


def cluster_articles(
    articles: list[Article],
    n_clusters: int | None = None,
) -> list[Cluster]:
    """Кластеризует публикации и присваивает каждой теме название."""
    if not articles:
        return []

    n_clusters = n_clusters or settings.N_CLUSTERS
    texts = [a.text for a in articles]
    embeddings = embed_texts(texts)

    if len(articles) <= 2 or n_clusters <= 1:
        k = 1
    else:
        k = _choose_n_clusters(embeddings, n_clusters)

    logger.info("Кластеризация: k=%d для %d публикаций", k, len(articles))

    if k == 1:
        labels = np.zeros(len(articles), dtype=int)
    else:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(embeddings)

    groups: dict[int, list[Article]] = {}
    for article, label in zip(articles, labels):
        article.cluster_id = int(label)
        groups.setdefault(int(label), []).append(article)

    clusters: list[Cluster] = []
    for cluster_id, items in sorted(groups.items()):
        items.sort(key=lambda a: a.score, reverse=True)
        title = _generate_cluster_title(items)
        for a in items:
            a.cluster_title = title
        clusters.append(Cluster(cluster_id=cluster_id, title=title, articles=items))

    clusters.sort(key=lambda c: c.size, reverse=True)
    return clusters