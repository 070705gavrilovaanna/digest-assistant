"""Обертка над sentence-transformers для получения эмбеддингов."""
from __future__ import annotations

import logging
import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import settings

logger = logging.getLogger(__name__)

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """ загрузка модели эмбеддингов."""
    global _model
    if _model is None:
        logger.info("Загрузка модели эмбеддингов: %s", settings.EMBEDDING_MODEL)
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Возвращает нормализованные эмбеддинги для списка текстов."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    model = get_model()
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return embeddings


def embed_one(text: str) -> np.ndarray:
    """Эмбеддинг одного текста, форма (dim,)."""
    return embed_texts([text])[0]