"""Тесты экспорта дайджеста."""
from __future__ import annotations

from pathlib import Path

from src.exporter import export_docx, export_markdown
from src.models import Article, Cluster, Digest


def _sample_digest() -> Digest:
    a1 = Article(
        title="Уязвимость в SCADA",
        link="https://example.com/1",
        source="Kaspersky ICS CERT",
        summary="Краткое описание.",
        score=0.85,
        llm_summary="Резюме от LLM.",
    )
    a2 = Article(
        title="Атака на АСУ ТП",
        link="https://example.com/2",
        source="Habr",
        summary="Ещё одно описание.",
        score=0.72,
        llm_summary="Другое резюме.",
    )
    cluster = Cluster(cluster_id=0, title="Промышленная ИБ", articles=[a1, a2])
    return Digest(
        interests=["КИИ", "АСУ ТП"],
        clusters=[cluster],
        total_articles=100,
        selected_articles=2,
    )


def test_markdown_contains_key_parts():
    digest = _sample_digest()
    md = digest.to_markdown()
    assert "ИТ/ИБ-дайджест" in md
    assert "Промышленная ИБ" in md
    assert "Уязвимость в SCADA" in md
    assert "https://example.com/1" in md


def test_export_markdown_creates_file(tmp_path: Path):
    digest = _sample_digest()
    out = tmp_path / "digest.md"
    export_markdown(digest, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_export_docx_creates_file(tmp_path: Path):
    digest = _sample_digest()
    out = tmp_path / "digest.docx"
    export_docx(digest, out)
    assert out.exists()
    assert out.stat().st_size > 0