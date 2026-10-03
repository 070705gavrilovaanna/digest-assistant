"""Тесты ранжирования и diversity reranking."""
from __future__ import annotations

from src.models import Article
from src.ranker import _diversity_rerank, _minmax_norm

import numpy as np


def _article(title: str, source: str, score: float) -> Article:
    a = Article(title=title, link=f"https://x/{title}", source=source)
    a.score = score
    return a


def test_minmax_norm_basic():
    arr = np.array([0.0, 5.0, 10.0])
    norm = _minmax_norm(arr)
    assert norm[0] == 0.0
    assert norm[2] == 1.0
    assert abs(norm[1] - 0.5) < 1e-6


def test_minmax_norm_constant():
    arr = np.array([3.0, 3.0, 3.0])
    norm = _minmax_norm(arr)
    assert (norm == 0).all()


def test_diversity_rerank_limits_sources():
    ranked = [
        _article("a1", "CISA", 0.9),
        _article("a2", "CISA", 0.8),
        _article("a3", "CISA", 0.7),
        _article("a4", "CISA", 0.6),
        _article("b1", "Habr", 0.5),
        _article("b2", "Habr", 0.4),
        _article("c1", "Kaspersky", 0.3),
    ]
    result = _diversity_rerank(ranked, top_k=5, max_per_source=2)
    # первые 2 должны быть из CISA, потом Habr, потом Kaspersky
    sources = [a.source for a in result]
    assert sources.count("CISA") <= 2
    assert "Habr" in sources
    assert "Kaspersky" in sources
    assert len(result) == 5


def test_diversity_rerank_backfills():
    ranked = [
        _article("a1", "CISA", 0.9),
        _article("a2", "CISA", 0.8),
        _article("a3", "CISA", 0.7),
        _article("a4", "CISA", 0.6),
    ]
    # Просим 4, но лимит 2 на источник. После diversity у нас 2,
    # поэтому должны добрать из отложенных.
    result = _diversity_rerank(ranked, top_k=4, max_per_source=2)
    assert len(result) == 4