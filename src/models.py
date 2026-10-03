"""Модели данных проекта."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Article:
    """Одна публикация."""

    title: str
    link: str
    source: str
    summary: str = ""
    published: str = ""
    tags: list[str] = field(default_factory=list)

    # Заполняется на этапе ранжирования
    score: float = 0.0
    embedding_score: float = 0.0
    bm25_score: float = 0.0

    # Заполняется на этапе кластеризации и суммаризации
    cluster_id: Optional[int] = None
    cluster_title: str = ""
    llm_summary: str = ""

    # Уникальный идентификатор для дедупликации
    uid: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Article":
        allowed = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in data.items() if k in allowed})

    @property
    def text(self) -> str:
        """Текст публикации, склеенный для эмбеддингов и BM25."""
        parts = [self.title or "", self.summary or ""]
        return " ".join(p for p in parts if p).strip()


@dataclass
class Cluster:
    """Тематический кластер публикаций."""

    cluster_id: int
    title: str
    articles: list[Article] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.articles)


@dataclass
class Digest:
    """Готовый дайджест."""

    interests: list[str]
    clusters: list[Cluster]
    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )
    total_articles: int = 0
    selected_articles: int = 0

    def to_markdown(self) -> str:
        lines: list[str] = []
        lines.append("# ИТ/ИБ-дайджест")
        lines.append("")
        lines.append(f"**Интересы:** {', '.join(self.interests)}")
        lines.append(f"**Дата формирования:** {self.created_at}")
        lines.append(
            f"**Всего собрано:** {self.total_articles}  "
            f"**Отобрано:** {self.selected_articles}"
        )
        lines.append("")
        lines.append("---")
        lines.append("")

        for cluster in self.clusters:
            lines.append(f"## {cluster.title} ({cluster.size})")
            lines.append("")
            for a in cluster.articles:
                lines.append(f"### [{a.title}]({a.link})")
                lines.append(
                    f"*Источник: {a.source}  |  Релевантность: {a.score:.2f}*"
                )
                lines.append("")
                body = a.llm_summary or a.summary
                if body:
                    lines.append(body.strip())
                    lines.append("")
            lines.append("---")
            lines.append("")

        return "\n".join(lines)