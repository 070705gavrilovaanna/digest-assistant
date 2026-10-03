"""Гибридное ранжирование публикаций с diversity reranking.

Итоговая оценка: взвешенная сумма косинусной близости эмбеддингов
и нормализованного BM25-скора. Дополнительно ограничиваем число
публикаций от одного источника в топе, чтобы дайджест был разнообразным.
"""
from __future__ import annotations

import logging

import numpy as np
from rank_bm25 import BM25Okapi

from src.config import settings
from src.embedder import embed_texts
from src.models import Article

logger = logging.getLogger(__name__)

# Максимум публикаций из одного источника в итоговом топе
MAX_PER_SOURCE = 3


def _tokenize(text: str) -> list[str]:
    return [t for t in text.lower().split() if len(t) > 1]


def _minmax_norm(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    vmin, vmax = float(values.min()), float(values.max())
    if vmax - vmin < 1e-9:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def _diversity_rerank(
    ranked: list[Article],
    top_k: int,
    max_per_source: int,
) -> list[Article]:
    """Ограничивает число публикаций от одного источника в топе.

    Проходим по ranked сверху вниз, набираем top_k, но не берём больше
    max_per_source статей из одного источника, пока есть альтернативы.
    """
    selected: list[Article] = []
    deferred: list[Article] = []
    source_counts: dict[str, int] = {}

    for article in ranked:
        if len(selected) >= top_k:
            break
        count = source_counts.get(article.source, 0)
        if count < max_per_source:
            selected.append(article)
            source_counts[article.source] = count + 1
        else:
            deferred.append(article)

    # Если после прохода недобрали top_k, добираем из отложенных
    for article in deferred:
        if len(selected) >= top_k:
            break
        selected.append(article)

    return selected


def rank_articles(
    articles: list[Article],
    interests: list[str],
    top_k: int | None = None,
    max_per_source: int = MAX_PER_SOURCE,
) -> list[Article]:
    """Ранжирует публикации и возвращает top_k наиболее релевантных."""
    if not articles or not interests:
        return []

    top_k = top_k or settings.TOP_K_ARTICLES

    # 1. Эмбеддинги
    texts = [a.text for a in articles]
    article_embs = embed_texts(texts)

    profile_text = " ".join(interests)
    profile_emb = embed_texts([profile_text])[0]

    cosine_scores = article_embs @ profile_emb
    cosine_scores = np.asarray(cosine_scores, dtype=np.float32)

    # 2. BM25
    tokenized_corpus = [_tokenize(t) for t in texts]
    tokenized_query = _tokenize(profile_text)

    if tokenized_query and any(tokenized_corpus):
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores = np.asarray(bm25.get_scores(tokenized_query), dtype=np.float32)
    else:
        bm25_scores = np.zeros(len(articles), dtype=np.float32)

    # 3. Нормализация и смешивание
    cosine_norm = _minmax_norm(cosine_scores)
    bm25_norm = _minmax_norm(bm25_scores)

    final = (
        settings.EMBEDDING_WEIGHT * cosine_norm
        + settings.BM25_WEIGHT * bm25_norm
    )

    for article, emb_score, bm25_score, final_score in zip(
        articles, cosine_scores, bm25_scores, final
    ):
        article.embedding_score = float(emb_score)
        article.bm25_score = float(bm25_score)
        article.score = float(final_score)

    ranked = sorted(articles, key=lambda a: a.score, reverse=True)

    # 4. Diversity reranking
    result = _diversity_rerank(ranked, top_k=top_k, max_per_source=max_per_source)

    logger.info(
        "Ранжирование: из %d публикаций отобрано %d "
        "(лимит %d на источник)",
        len(articles),
        len(result),
        max_per_source,
    )
    return result