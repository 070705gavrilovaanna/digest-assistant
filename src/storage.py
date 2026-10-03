"""Хранилище публикаций на SQLite."""
from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from typing import Iterator

from src.config import DB_PATH
from src.models import Article

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    uid TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    link TEXT NOT NULL,
    source TEXT NOT NULL,
    summary TEXT,
    published TEXT,
    tags TEXT,
    score REAL DEFAULT 0,
    embedding_score REAL DEFAULT 0,
    bm25_score REAL DEFAULT 0,
    cluster_id INTEGER,
    cluster_title TEXT,
    llm_summary TEXT
);
CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source);
CREATE INDEX IF NOT EXISTS idx_articles_score ON articles(score);
"""


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(SCHEMA)
    logger.info("База данных инициализирована: %s", DB_PATH)


def _row_to_article(row: sqlite3.Row) -> Article:
    tags_raw = row["tags"] or "[]"
    try:
        tags = json.loads(tags_raw)
    except json.JSONDecodeError:
        tags = []
    return Article(
        uid=row["uid"],
        title=row["title"],
        link=row["link"],
        source=row["source"],
        summary=row["summary"] or "",
        published=row["published"] or "",
        tags=tags,
        score=row["score"] or 0.0,
        embedding_score=row["embedding_score"] or 0.0,
        bm25_score=row["bm25_score"] or 0.0,
        cluster_id=row["cluster_id"],
        cluster_title=row["cluster_title"] or "",
        llm_summary=row["llm_summary"] or "",
    )


def save_articles(articles: list[Article]) -> int:
    """Сохраняет или обновляет публикации. Возвращает число строк."""
    if not articles:
        return 0
    with _connect() as conn:
        conn.executemany(
            """
            INSERT INTO articles (
                uid, title, link, source, summary, published, tags,
                score, embedding_score, bm25_score,
                cluster_id, cluster_title, llm_summary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(uid) DO UPDATE SET
                title = excluded.title,
                link = excluded.link,
                source = excluded.source,
                summary = excluded.summary,
                published = excluded.published,
                tags = excluded.tags,
                score = excluded.score,
                embedding_score = excluded.embedding_score,
                bm25_score = excluded.bm25_score,
                cluster_id = excluded.cluster_id,
                cluster_title = excluded.cluster_title,
                llm_summary = excluded.llm_summary
            """,
            [
                (
                    a.uid,
                    a.title,
                    a.link,
                    a.source,
                    a.summary,
                    a.published,
                    json.dumps(a.tags, ensure_ascii=False),
                    a.score,
                    a.embedding_score,
                    a.bm25_score,
                    a.cluster_id,
                    a.cluster_title,
                    a.llm_summary,
                )
                for a in articles
            ],
        )
    return len(articles)


def load_all_articles() -> list[Article]:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM articles").fetchall()
    return [_row_to_article(r) for r in rows]


def count_articles() -> int:
    with _connect() as conn:
        row = conn.execute("SELECT COUNT(*) AS c FROM articles").fetchone()
    return int(row["c"]) if row else 0


def clear() -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM articles")
    logger.info("Таблица articles очищена")