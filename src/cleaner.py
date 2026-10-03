"""Очистка текста публикаций, дедупликация и фильтрация по дате."""
from __future__ import annotations

import hashlib
import html
import logging
import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse, urlunparse

from bs4 import BeautifulSoup

from src.models import Article

logger = logging.getLogger(__name__)

WHITESPACE_RE = re.compile(r"\s+")
TAG_RE = re.compile(r"<[^>]+>")

MAX_AGE_DAYS = 180

# Сколько первых значимых слов заголовка считаем сигнатурой
NEAR_DUP_PREFIX_WORDS = 4


# стоп-слова и числительные для сигнатур почти-дублей

_SIGNATURE_STOPWORDS = {
    "a", "an", "the", "of", "to", "in", "on", "for", "and", "or",
    "with", "by", "at", "from", "as", "is", "are", "be", "was",
    "adds", "add", "added", "new", "known", "exploited", "update",
    "updates", "release", "released", "announces", "announced",
    "и", "в", "на", "с", "по", "для", "из", "к", "о", "об", "от",
    "при", "за", "над", "под", "до", "после", "или", "но", "же",
}

_SIGNATURE_NUMERALS = {
    "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve",
    "один", "одна", "одно", "два", "две", "три", "четыре", "пять",
    "шесть", "семь", "восемь", "девять", "десять",
}

_CVE_RE = re.compile(r"cve[-_]\d{4}[-_]\d{3,6}", re.IGNORECASE)


# базовые операции 

def strip_html(text: str) -> str:
    """Удаляет HTML-теги и декодирует сущности."""
    if not text:
        return ""
    if "<" in text and ">" in text:
        try:
            text = BeautifulSoup(text, "lxml").get_text(separator=" ")
        except Exception:
            text = TAG_RE.sub(" ", text)
    return html.unescape(text)


def normalize_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def normalize_url(url: str) -> str:
    """Убирает UTM-метки и фрагменты из URL."""
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        clean_query = "&".join(
            part for part in parsed.query.split("&")
            if not part.lower().startswith(("utm_", "fbclid", "gclid"))
        )
        return urlunparse(
            (parsed.scheme, parsed.netloc, parsed.path, parsed.params, clean_query, "")
        )
    except Exception:
        return url


def compute_uid(article: Article) -> str:
    """Уникальный идентификатор публикации по ссылке и заголовку."""
    key = f"{normalize_url(article.link)}|{article.title.lower().strip()}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


def clean_article(article: Article) -> Article:
    """Очищает одну публикацию."""
    article.title = normalize_whitespace(strip_html(article.title))
    article.summary = normalize_whitespace(strip_html(article.summary))
    article.link = normalize_url(article.link)
    article.uid = compute_uid(article)
    return article


def clean_articles(articles: list[Article]) -> list[Article]:
    cleaned = [clean_article(a) for a in articles]
    logger.info("Очищено публикаций: %d", len(cleaned))
    return cleaned


def deduplicate(articles: list[Article]) -> list[Article]:
    """Удаляет полные дубликаты по uid."""
    seen: set[str] = set()
    unique: list[Article] = []
    for a in articles:
        if not a.uid:
            a.uid = compute_uid(a)
        if a.uid in seen:
            continue
        seen.add(a.uid)
        unique.append(a)
    logger.info(
        "Дедупликация: было %d, осталось %d", len(articles), len(unique)
    )
    return unique


# сигнатуры и почти-дубли

def _stem_word(word: str) -> str:
    """Грубое усечение английских окончаний для сигнатур заголовков.

    Не настоящий стеммер, но для поиска почти-дублей этого достаточно:
    vulnerabilities -> vulnerability, bugs -> bug, patches -> patch.
    """
    if len(word) < 5:
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("es") and len(word) > 5:
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _title_signature(title: str, n_words: int = NEAR_DUP_PREFIX_WORDS) -> str:
    """Сигнатура заголовка для поиска почти-дублей.

    Шаги:
        1. Убираем CVE-идентификаторы.
        2. Разбиваем на слова.
        3. Отбрасываем стоп-слова и числительные.
        4. Применяем грубое усечение окончаний.
        5. Берём первые n_words слов.

    Это позволяет ловить серии заголовков вида
    "CISA Adds One Known Exploited Vulnerability ..." и
    "CISA Adds Two Known Exploited Vulnerabilities ..." как почти-дубли.
    """
    text = _CVE_RE.sub(" ", title.lower())
    words = re.findall(r"[а-яёa-z][а-яёa-z\-]+", text)

    filtered: list[str] = []
    for w in words:
        if w in _SIGNATURE_STOPWORDS or w in _SIGNATURE_NUMERALS:
            continue
        filtered.append(_stem_word(w))

    return " ".join(filtered[:n_words])


def drop_near_duplicates(articles: list[Article]) -> list[Article]:
    """Удаляет почти-дубли внутри одного источника.

    Две статьи считаются почти-дублями, если:
        - они из одного источника,
        - сигнатуры заголовков совпадают (без учёта стоп-слов,
          числительных и CVE-идентификаторов).

    Из группы почти-дублей оставляем самую свежую.
    """
    buckets: dict[tuple[str, str], list[Article]] = {}
    for a in articles:
        sig = _title_signature(a.title)
        if not sig:
            buckets[(a.source, a.uid)] = [a]
            continue
        buckets.setdefault((a.source, sig), []).append(a)

    kept: list[Article] = []
    removed = 0
    for group in buckets.values():
        if len(group) == 1:
            kept.append(group[0])
            continue

        def sort_key(a: Article):
            dt = parse_date(a.published)
            return (dt is None, -(dt.timestamp() if dt else 0))

        group.sort(key=sort_key)
        kept.append(group[0])
        removed += len(group) - 1

    logger.info(
        "Удаление почти-дублей: было %d, осталось %d (убрано %d)",
        len(articles),
        len(kept),
        removed,
    )
    return kept


# язык и даты

def is_relevant_language(article: Article) -> bool:
    text = article.text
    if not text:
        return False
    has_cyrillic = any("а" <= ch.lower() <= "я" for ch in text)
    has_latin = any("a" <= ch.lower() <= "z" for ch in text)
    return has_cyrillic or has_latin


def parse_date(raw: str) -> datetime | None:
    """Парсит дату из строки RSS. Возвращает None, если не получилось."""
    if not raw:
        return None
    raw = raw.strip()

    try:
        dt = parsedate_to_datetime(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        pass

    for fmt in (
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            dt = datetime.strptime(raw, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue

    return None


def filter_by_freshness(
    articles: list[Article],
    max_age_days: int = MAX_AGE_DAYS,
) -> list[Article]:
    """Оставляет публикации не старше max_age_days."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    kept: list[Article] = []
    dropped_old = 0
    no_date = 0

    for a in articles:
        dt = parse_date(a.published)
        if dt is None:
            no_date += 1
            kept.append(a)
            continue
        if dt >= cutoff:
            kept.append(a)
        else:
            dropped_old += 1

    logger.info(
        "Фильтр по свежести: оставлено %d, отброшено старых %d, без даты %d",
        len(kept),
        dropped_old,
        no_date,
    )
    return kept