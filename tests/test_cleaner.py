"""Тесты модуля очистки публикаций."""
from __future__ import annotations

from src.cleaner import (
    clean_article,
    compute_uid,
    deduplicate,
    drop_near_duplicates,
    filter_by_freshness,
    normalize_url,
    normalize_whitespace,
    parse_date,
    strip_html,
)
from src.models import Article


def _make_article(title: str, link: str = "", source: str = "test",
                  published: str = "") -> Article:
    return Article(title=title, link=link, source=source, published=published)


def test_strip_html_removes_tags():
    raw = "<p>Привет, <b>мир</b>!</p>"
    # strip_html может вставить лишние пробелы на месте тегов,
    # normalize_whitespace их сжимает в один
    result = normalize_whitespace(strip_html(raw))
    assert result == "Привет, мир !"


def test_normalize_url_strips_utm():
    url = "https://example.com/news?utm_source=habr&id=42"
    cleaned = normalize_url(url)
    assert "utm_source" not in cleaned
    assert "id=42" in cleaned


def test_compute_uid_is_stable():
    a = _make_article("Заголовок", "https://example.com/a")
    uid1 = compute_uid(a)
    uid2 = compute_uid(a)
    assert uid1 == uid2
    assert len(uid1) == 24


def test_deduplicate_removes_same_articles():
    a1 = _make_article("Новость", "https://example.com/a")
    a2 = _make_article("Новость", "https://example.com/a")
    a3 = _make_article("Другая", "https://example.com/b")
    for a in (a1, a2, a3):
        clean_article(a)
    unique = deduplicate([a1, a2, a3])
    assert len(unique) == 2


def test_drop_near_duplicates_same_prefix():
    a1 = _make_article(
        "CISA Adds One Known Exploited Vulnerability",
        "https://cisa.gov/1",
        source="CISA",
        published="Fri, 26 Sep 2026 10:00:00 GMT",
    )
    a2 = _make_article(
        "CISA Adds Two Known Exploited Vulnerabilities",
        "https://cisa.gov/2",
        source="CISA",
        published="Thu, 25 Sep 2026 10:00:00 GMT",
    )
    a3 = _make_article(
        "Другая новость про Siemens",
        "https://cisa.gov/3",
        source="CISA",
        published="Fri, 26 Sep 2026 10:00:00 GMT",
    )
    kept = drop_near_duplicates([a1, a2, a3])
    # a1 и a2 после удаления числительных и стоп-слов дают
    # одинаковую сигнатуру "cisa vulnerability"
    assert len(kept) == 2
    # из пары оставляем самую свежую - a1 (26 Sep > 25 Sep)
    assert any(a.link == "https://cisa.gov/1" for a in kept)


def test_drop_near_duplicates_ignores_cve_ids():
    a1 = _make_article(
        "CISA Adds Known Exploited Vulnerability CVE-2026-11111",
        "https://cisa.gov/1",
        source="CISA",
        published="Fri, 26 Sep 2026 10:00:00 GMT",
    )
    a2 = _make_article(
        "CISA Adds Known Exploited Vulnerability CVE-2026-22222",
        "https://cisa.gov/2",
        source="CISA",
        published="Thu, 25 Sep 2026 10:00:00 GMT",
    )
    kept = drop_near_duplicates([a1, a2])
    assert len(kept) == 1
    assert kept[0].link == "https://cisa.gov/1"


def test_parse_date_iso():
    dt = parse_date("2026-09-29T16:11:26Z")
    assert dt is not None
    assert dt.year == 2026 and dt.month == 9


def test_filter_by_freshness_drops_old():
    fresh = _make_article(
        "Свежая", "https://example.com/fresh",
        published="2026-09-29T10:00:00Z",
    )
    old = _make_article(
        "Старая", "https://example.com/old",
        published="2018-01-01T10:00:00Z",
    )
    kept = filter_by_freshness([fresh, old], max_age_days=180)
    assert len(kept) == 1
    assert kept[0].title == "Свежая"