"""
web_search.py
--------------
Lightweight, keyless web-search augmentation for the Veronica / Noah chatbot.

Uses DuckDuckGo (via the `ddgs` package, formerly `duckduckgo_search`) for
free web search - no API key required. Optionally fetches the top result
pages for richer context (snippets alone are often too thin), then formats
everything into a single prompt-ready block that gets handed to the LLM.

Install:
    pip install ddgs beautifulsoup4

(If you already have the older `duckduckgo_search` package installed,
this module will fall back to importing from that instead.)
"""

import re
import time
import logging
import requests
from bs4 import BeautifulSoup

try:
    from ddgs import DDGS  # current package name
except ImportError:
    from duckduckgo_search import DDGS  # fallback for older installs

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; NoahBot/1.0; +https://byncai.net)"
}

# Words/phrases that hint the user wants fresh, real-world info.
# This runs BEFORE we decide whether to hit the web, so it's fine for it
# to be a little generous - a false positive just means we search a bit
# more often than strictly necessary, which is harmless.
WEB_SEARCH_TRIGGERS = [
    "latest", "today", "current", "currently", "news", "update", "updated",
    "right now", "this week", "this month", "this year", "recent", "recently",
    "price of", "stock", "score", "weather", "who is the", "who won",
    "release date", "when is", "when was", "what happened", "happening now",
    "election", "ceo of", "president of",
]


def needs_web_search(question: str) -> bool:
    """Cheap heuristic: does this question likely need live web info?"""
    q = question.lower()
    if any(trigger in q for trigger in WEB_SEARCH_TRIGGERS):
        return True
    # A "current-ish" year (2024-2029) is a strong signal too.
    if re.search(r"\b(202[4-9])\b", q):
        return True
    return False


def search_web(query: str, max_results: int = 5):
    """
    Run a DuckDuckGo text search.
    Returns a list of {title, url, snippet} dicts, or [] on failure -
    callers should treat an empty list as "no web context available"
    and just let the LLM answer from its own knowledge / RAG chunks.
    """
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href") or r.get("url", ""),
                    "snippet": r.get("body", ""),
                })
    except Exception as e:
        logger.exception("web search failed for query=%r: %s", query, e)
    return results


def fetch_page_text(url: str, max_chars: int = 1500) -> str:
    """
    Best-effort fetch + strip of a result page's visible text, used to
    enrich thin snippets. Never raises - returns "" on any failure
    (timeout, 403, non-HTML content, etc.).
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=6)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "form"]):
            tag.decompose()

        text = " ".join(soup.get_text(separator=" ").split())
        return text[:max_chars]
    except Exception as e:
        logger.debug("page fetch failed for %s: %s", url, e)
        return ""



# --------------------
# Tiny TTL cache
# --------------------
# Since this now runs on essentially every non-KB message, an identical
# or near-identical question arriving again within a short window (a user
# repeating themselves, a page refresh, two people asking the same thing)
# shouldn't re-hit DuckDuckGo. Keeps things fast and avoids rate limits.
_CACHE: dict = {}
_CACHE_TTL_SECONDS = 10 * 60  # 10 minutes


def _cache_get(key: str):
    entry = _CACHE.get(key)
    if not entry:
        return None
    value, expires_at = entry
    if time.time() > expires_at:
        _CACHE.pop(key, None)
        return None
    return value


def _cache_set(key: str, value: str) -> None:
    _CACHE[key] = (value, time.time() + _CACHE_TTL_SECONDS)
    # Cheap cleanup so this dict doesn't grow forever in a long-running process
    if len(_CACHE) > 500:
        now = time.time()
        for k, (_, exp) in list(_CACHE.items()):
            if now > exp:
                _CACHE.pop(k, None)


def build_web_context(query: str, max_results: int = 5, fetch_pages: bool = True,
                       max_pages_to_fetch: int = 3) -> str:
    """
    Search the web and assemble one context block, formatted so it's easy
    for the LLM to reference naturally:

        [1] Title — url
        snippet or fetched page text

        [2] Title — url
        ...

    Returns "" if the search failed or returned nothing, so the caller
    can gracefully skip adding web context to the prompt.
    """
    cache_key = query.strip().lower()
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    results = search_web(query, max_results=max_results)
    if not results:
        return ""

    blocks = []
    for i, r in enumerate(results, start=1):
        body = r["snippet"]
        if fetch_pages and i <= max_pages_to_fetch:
            page_text = fetch_page_text(r["url"])
            if page_text:
                body = page_text  # richer than the raw snippet

        if not body:
            continue

        blocks.append(f"[{i}] {r['title']} — {r['url']}\n{body}")

    context = "\n\n".join(blocks)
    _cache_set(cache_key, context)
    return context
