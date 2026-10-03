"""CLI-версия пайплайна без интерфейса Streamlit."""
from __future__ import annotations

import argparse
import logging
from datetime import datetime

from src.cleaner import clean_articles, deduplicate, drop_near_duplicates, is_relevant_language, filter_by_freshness, is_relevant_language
from src.clusterer import cluster_articles
from src.collector import collect_all
from src.config import EXAMPLES_DIR, settings
from src.exporter import export_docx, export_markdown
from src.models import Digest
from src.ranker import rank_articles
from src.storage import init_db, save_articles
from src.summarizer import summarize_all

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ИТ/ИБ-дайджест (CLI)")
    parser.add_argument(
        "--interests",
        type=str,
        default=", ".join(settings.DEFAULT_INTERESTS),
        help="Интересы через запятую",
    )
    parser.add_argument("--top-k", type=int, default=settings.TOP_K_ARTICLES)
    parser.add_argument("--clusters", type=int, default=settings.N_CLUSTERS)
    parser.add_argument(
        "--no-llm-summary",
        action="store_true",
        help="Отключить суммаризацию через LLM",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    interests = [i.strip() for i in args.interests.split(",") if i.strip()]
    if not interests:
        print("Не заданы интересы.")
        return

    init_db()

    print("Сбор публикаций...")
    raw = collect_all()
    print(f"Собрано: {len(raw)}")
    if not raw:
        return

    cleaned = clean_articles(raw)
    cleaned = [a for a in cleaned if is_relevant_language(a)]
    cleaned = filter_by_freshness(cleaned, max_age_days=180)
    unique = deduplicate(cleaned)
    unique = drop_near_duplicates(unique)
    print(f"Уникальных публикаций: {len(unique)}")

    selected = rank_articles(unique, interests, top_k=args.top_k)
    print(f"Отобрано по интересам: {len(selected)}")

    clusters = cluster_articles(selected, n_clusters=args.clusters)
    print(f"Тем: {len(clusters)}")

    if not args.no_llm_summary:
        print("Суммаризация...")
        summarize_all(clusters)

    save_articles(selected)

    digest = Digest(
        interests=interests,
        clusters=clusters,
        total_articles=len(raw),
        selected_articles=len(selected),
    )

    suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = EXAMPLES_DIR / f"digest_{suffix}.md"
    docx_path = EXAMPLES_DIR / f"digest_{suffix}.docx"
    export_markdown(digest, md_path)
    export_docx(digest, docx_path)

    print(f"Markdown: {md_path}")
    print(f"DOCX: {docx_path}")


if __name__ == "__main__":
    main()