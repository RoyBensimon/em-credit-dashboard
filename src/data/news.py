"""
Market News Aggregator
======================
Fetches latest financial, macro, and geopolitical news from RSS feeds.

Architecture
------------
- Pure stdlib + certifi (SSL fix for macOS). No extra dependencies.
- Module-level TTL cache: network calls happen at most once per CACHE_TTL_SECONDS.
  Every Streamlit rerun within the TTL window returns instantly from memory.
- All sources fetched concurrently; one failure never blocks the others.
- NO restrictive keyword filtering — all articles from financial sources are kept.

Adding / removing sources
--------------------------
Edit RSS_SOURCES. Each entry:
  name   — display label shown in the UI
  url    — any RSS 2.0 or Atom feed URL
  region — optional tag for badge display (can be "")
No other code changes required.

Debugging
---------
Set LOG_LEVEL = logging.DEBUG to see per-source fetch counts in the terminal.
"""

from __future__ import annotations

import logging
import re
import ssl
import time
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Optional

# ── Logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [news] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

# ── SSL context (fixes macOS certificate errors) ──────────────────────────────
# certifi ships with most Python environments. Falls back to system CA if absent.
def _make_ssl_ctx() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()

_SSL_CTX = _make_ssl_ctx()

# ── Cache settings ────────────────────────────────────────────────────────────
CACHE_TTL_SECONDS = 60  # 1 minute

_cache: dict = {"data": [], "fetched_at": 0.0}

# ── RSS Sources ───────────────────────────────────────────────────────────────
# All verified working (tested 2025). All free, no auth required.
#
# Sources:
#   1. Google News Finance — broad financial & macro coverage, very reliable
#   2. Google News Macro   — central bank, geopolitics, economy focus
#   3. CNBC Markets        — US and global markets, high quality
#   4. BBC Business        — global business & geopolitical news
#   5. FT Markets          — premium financial content (free RSS tier)
#
# To add a source: append a dict below. To remove: delete the entry.

RSS_SOURCES: list[dict] = [
    {
        "name": "Google News",
        "url": (
            "https://news.google.com/rss/search"
            "?q=finance+markets+economy&hl=en-US&gl=US&ceid=US:en"
        ),
    },
    {
        "name": "Google News",
        "url": (
            "https://news.google.com/rss/search"
            "?q=central+bank+geopolitics+economy&hl=en-US&gl=US&ceid=US:en"
        ),
    },
    {
        "name": "CNBC",
        "url":  "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    },
    {
        "name": "BBC Business",
        "url":  "https://feeds.bbci.co.uk/news/business/rss.xml",
    },
    {
        "name": "FT Markets",
        "url":  "https://www.ft.com/markets?format=rss",
    },
]

# ── Country detection (for badge display only — does NOT filter articles) ─────
_COUNTRIES: list[str] = [
    "Brazil", "Mexico", "Argentina", "Israel", "Turkey", "South Africa",
    "Egypt", "Saudi Arabia", "UAE", "Qatar", "Nigeria", "Kenya",
    "India", "Indonesia", "Vietnam", "Philippines", "Thailand",
    "Malaysia", "China", "Colombia", "Chile", "Peru",
    "Poland", "Hungary", "Romania", "Ukraine",
    "Russia", "Iran", "Japan", "South Korea", "Germany", "France",
    "United Kingdom", "UK", "United States", "US",
]

_COUNTRY_REGION: dict[str, str] = {
    "Brazil": "LatAm",    "Mexico": "LatAm",       "Argentina": "LatAm",
    "Colombia": "LatAm",  "Chile": "LatAm",         "Peru": "LatAm",
    "Turkey": "CEEMEA",   "South Africa": "CEEMEA", "Egypt": "CEEMEA",
    "Saudi Arabia": "CEEMEA", "UAE": "CEEMEA",      "Qatar": "CEEMEA",
    "Nigeria": "CEEMEA",  "Kenya": "CEEMEA",        "Israel": "CEEMEA",
    "Poland": "CEEMEA",   "Hungary": "CEEMEA",      "Romania": "CEEMEA",
    "Ukraine": "CEEMEA",  "Russia": "CEEMEA",       "Iran": "CEEMEA",
    "India": "Asia",      "Indonesia": "Asia",       "Vietnam": "Asia",
    "Philippines": "Asia","Thailand": "Asia",        "Malaysia": "Asia",
    "China": "Asia",      "Japan": "Asia",           "South Korea": "Asia",
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _fetch_raw(url: str, timeout: int = 8) -> Optional[str]:
    """HTTP GET with certifi SSL context. Returns None on any error."""
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; MarketNewsFeed/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as resp:
            content = resp.read().decode("utf-8", errors="replace")
        logger.debug("Fetched %s — %d bytes", url[:60], len(content))
        return content
    except Exception as exc:
        logger.warning("Fetch failed for %s: %s", url[:60], exc)
        return None


def _parse_date(raw: str) -> datetime:
    """Parse RFC 2822 pubDate to UTC datetime. Falls back to now on error."""
    if not raw:
        return datetime.now(tz=timezone.utc)
    try:
        return parsedate_to_datetime(raw).astimezone(timezone.utc)
    except Exception:
        return datetime.now(tz=timezone.utc)


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


def _detect_country(text: str) -> tuple[str, str]:
    """Return (country, region) from the first matched country name, or ('', '')."""
    tl = text.lower()
    for country in _COUNTRIES:
        if country.lower() in tl:
            return country, _COUNTRY_REGION.get(country, "")
    return "", ""


def _parse_feed(xml_text: str, source_name: str) -> list[dict]:
    """Parse RSS 2.0 or Atom XML into normalized article dicts."""
    articles: list[dict] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.warning("XML parse error for %s: %s", source_name, exc)
        return articles

    ns    = {"atom": "http://www.w3.org/2005/Atom"}
    items = root.findall(".//item") or root.findall(".//atom:entry", ns)
    logger.info("  %s: %d raw items in feed", source_name, len(items))

    for item in items:
        def _t(tag: str, atom: bool = False) -> str:
            el = item.find(f"atom:{tag}", ns) if atom else item.find(tag)
            return _strip_html((el.text or "") if el is not None else "")

        title   = _t("title")
        url     = _t("link") or _t("guid")
        summary = _t("description") or _t("summary", atom=True)
        date    = _parse_date(
            _t("pubDate") or _t("updated", atom=True) or _t("published", atom=True)
        )

        if not title or not url:
            continue

        country, region = _detect_country(title + " " + summary)

        articles.append({
            "title":   title[:200],
            "url":     url,
            "source":  source_name,
            "date":    date,
            "summary": summary[:280] if summary else "",
            "country": country,
            "region":  region,
        })

    logger.info("  %s: %d articles parsed", source_name, len(articles))
    return articles


def _deduplicate(articles: list[dict]) -> list[dict]:
    """
    Two-pass deduplication:
      1. URL dedup (strip query params).
      2. Near-duplicate title (first 60 chars): keep entry with longer summary.
    """
    seen_urls:   set[str]       = set()
    seen_titles: dict[str, int] = {}
    result: list[dict] = []

    for art in articles:
        url_key   = art["url"].split("?")[0].rstrip("/")
        title_key = art["title"][:60].lower()

        if url_key in seen_urls:
            continue
        seen_urls.add(url_key)

        if title_key in seen_titles:
            idx = seen_titles[title_key]
            if len(art["summary"]) > len(result[idx]["summary"]):
                result[idx] = art
            continue

        seen_titles[title_key] = len(result)
        result.append(art)

    return result


def _fetch_all(max_per_source: int) -> list[dict]:
    """Fetch all sources concurrently; failures are silently skipped."""
    all_articles: list[dict] = []

    def _fetch_source(src: dict) -> list[dict]:
        xml = _fetch_raw(src["url"])
        if not xml:
            return []
        items = _parse_feed(xml, src["name"])
        return items[:max_per_source]

    with ThreadPoolExecutor(max_workers=len(RSS_SOURCES)) as pool:
        futures = [pool.submit(_fetch_source, src) for src in RSS_SOURCES]
        for future in as_completed(futures):
            try:
                batch = future.result()
                all_articles.extend(batch)
            except Exception as exc:
                logger.error("Unexpected error in fetch thread: %s", exc)

    logger.info("Total after merge: %d articles", len(all_articles))
    return all_articles


# ── Public API ────────────────────────────────────────────────────────────────

def is_cache_fresh() -> bool:
    """True if cached data exists and is within CACHE_TTL_SECONDS."""
    return bool(_cache["data"]) and (time.time() - _cache["fetched_at"]) < CACHE_TTL_SECONDS


def cache_age_minutes() -> float:
    """Minutes since last successful fetch. Returns 0 if never fetched."""
    if not _cache["fetched_at"]:
        return 0.0
    return (time.time() - _cache["fetched_at"]) / 60


def get_news(max_per_source: int = 25, max_total: int = 60) -> list[dict]:
    """
    Return latest market/financial news, using the module-level TTL cache.

    - Cache fresh  → returns instantly (< 1 ms).
    - Cache stale  → fetches all sources concurrently, updates cache.

    Returns list of dicts sorted by date desc:
        title, url, source, date (datetime UTC), summary, country, region
    """
    if is_cache_fresh():
        logger.debug("Cache hit — returning %d cached articles", len(_cache["data"]))
        return _cache["data"]

    logger.info("Cache miss — fetching all RSS sources…")
    articles = _fetch_all(max_per_source)
    articles = _deduplicate(articles)
    logger.info("After deduplication: %d articles", len(articles))
    articles.sort(key=lambda a: a["date"], reverse=True)
    articles = articles[:max_total]

    _cache["data"]       = articles
    _cache["fetched_at"] = time.time()
    logger.info("Cache updated with %d articles", len(articles))

    return articles
