"""Streamlit-интерфейс ИИ-помощника для ИТ/ИБ-дайджестов."""
from __future__ import annotations

import logging
from datetime import datetime

import streamlit as st

from src.cleaner import clean_articles, deduplicate, drop_near_duplicates, filter_by_freshness, is_relevant_language
from src.clusterer import cluster_articles
from src.collector import collect_all
from src.config import EXAMPLES_DIR, load_sources, settings
from src.exporter import export_docx, export_markdown
from src.llm_client import get_llm
from src.models import Digest
from src.ranker import rank_articles
from src.storage import init_db, save_articles
from src.summarizer import summarize_all

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

st.set_page_config(
    page_title="ИТ/ИБ-дайджест",
    page_icon=None,
    layout="wide",
)


def _timestamp_suffix() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _render_sidebar() -> None:
    st.sidebar.header("Настройки")

    sources = load_sources()
    st.sidebar.write(f"Доступно источников: {len(sources)}")
    with st.sidebar.expander("Список источников"):
        for s in sources:
            st.write(f"- {s.get('name')}")

    st.sidebar.markdown("---")
    st.sidebar.write("Активные интересы по умолчанию:")
    st.sidebar.write(", ".join(settings.DEFAULT_INTERESTS))

    st.sidebar.markdown("---")
    st.sidebar.write("Статус моделей LLM:")
    try:
        llm = get_llm()
        for m in llm.status():
            flag = "активна" if m["is_active"] else "ожидает"
            if not m["available"]:
                state = f"остывает {m['cooldown_in_sec']} сек"
            else:
                state = "доступна"
            st.sidebar.write(f"- {m['model']} | {state} | {flag}")
    except Exception as e:
        st.sidebar.warning(f"LLM недоступен: {e}")


def main() -> None:
    init_db()
    _render_sidebar()

    st.title("ИИ-помощник для ИТ/ИБ-дайджестов")
    st.caption(
        "Собирает публикации, отбирает их по интересам пользователя "
        "и формирует тематический дайджест."
    )

    default_interests = ", ".join(settings.DEFAULT_INTERESTS)
    interests_raw = st.text_area(
        "Интересы пользователя (через запятую)",
        value=default_interests,
        height=100,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        top_k = st.number_input(
            "Сколько публикаций отобрать",
            min_value=5,
            max_value=100,
            value=settings.TOP_K_ARTICLES,
            step=5,
        )
    with col2:
        n_clusters = st.number_input(
            "Максимум тем",
            min_value=1,
            max_value=10,
            value=settings.N_CLUSTERS,
        )
    with col3:
        use_llm_summary = st.checkbox("Суммаризация через LLM", value=True)

    run = st.button("Собрать дайджест", type="primary")

    if not run:
        st.info("Нажми кнопку, чтобы запустить сбор и обработку публикаций.")
        return

    interests = [i.strip() for i in interests_raw.split(",") if i.strip()]
    if not interests:
        st.error("Введи хотя бы один интерес.")
        return

    # 1. Сбор публикаций
    with st.spinner("Собираю публикации из источников..."):
        raw = collect_all()
    if not raw:
        st.error("Не удалось собрать ни одной публикации. Проверь источники.")
        return
    st.write(f"Собрано публикаций: {len(raw)}")

    # 2. Очистка, фильтр по свежести, дедупликация, удаление почти-дублей
    cleaned = clean_articles(raw)
    cleaned = [a for a in cleaned if is_relevant_language(a)]
    cleaned = filter_by_freshness(cleaned, max_age_days=180)
    unique = deduplicate(cleaned)
    unique = drop_near_duplicates(unique)
    st.write(f"После очистки и фильтрации: {len(unique)}")

    # 3. Ранжирование с diversity
    with st.spinner("Отбираю релевантные публикации..."):
        selected = rank_articles(unique, interests, top_k=int(top_k))
    st.write(f"Отобрано по интересам: {len(selected)}")

    # 4. Кластеризация
    with st.spinner("Группирую публикации по темам..."):
        clusters = cluster_articles(selected, n_clusters=int(n_clusters))
    st.write(f"Получено тем: {len(clusters)}")

    # 5. Суммаризация через LLM (опционально)
    if use_llm_summary:
        with st.spinner("Готовлю резюме публикаций..."):
            summarize_all(clusters)

    # 6. Сохранение в БД
    save_articles(selected)

    # 7. Сборка дайджеста
    digest = Digest(
        interests=interests,
        clusters=clusters,
        total_articles=len(raw),
        selected_articles=len(selected),
    )
    markdown_text = digest.to_markdown()

    # 8. Отображение результата
    st.subheader("Итоговый дайджест")
    st.markdown(markdown_text)

    # 9. Экспорт в файлы
    suffix = _timestamp_suffix()
    md_path = EXAMPLES_DIR / f"digest_{suffix}.md"
    docx_path = EXAMPLES_DIR / f"digest_{suffix}.docx"

    export_markdown(digest, md_path)
    export_docx(digest, docx_path)

    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button(
            "Скачать Markdown",
            data=markdown_text,
            file_name=f"digest_{suffix}.md",
            mime="text/markdown",
        )
    with col_b:
        with open(docx_path, "rb") as f:
            st.download_button(
                "Скачать DOCX",
                data=f.read(),
                file_name=f"digest_{suffix}.docx",
                mime=(
                    "application/vnd.openxmlformats-officedocument"
                    ".wordprocessingml.document"
                ),
            )

    st.caption(f"Файлы также сохранены в папке: {EXAMPLES_DIR}")


if __name__ == "__main__":
    main()