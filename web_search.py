"""
web_search.py
--------------
Multi-provider web-search augmentation for the Veronica / Noah chatbot.

Provider order (first success wins):
    1. Brave Search API   - paid/free-tier key, most reliable, best rate limits
    2. SearXNG             - free, self-hosted (or public instance) metasearch
    3. DuckDuckGo (ddgs)   - free, keyless, last-resort fallback

Every provider is wrapped so a failure (timeout, bad key, rate limit, empty
result, exception of any kind) just falls through to the next one. If all
three fail, search_web() returns [] and callers already treat that as
"no web context available".

Install:
    pip install ddgs beautifulsoup4 requests

Environment variables:
    BRAVE_API_KEY   - your Brave Search API subscription token
                       (get one at https://brave.com/search/api/)
    SEARXNG_URL     - base URL of a SearXNG instance, e.g.
                       "https://searx.example.com" or your self-hosted
                       instance. If unset, SearXNG is skipped.
    SEARXNG_API_KEY - optional, only needed if your instance requires auth

If BRAVE_API_KEY / SEARXNG_URL are not set, those providers are silently
skipped and the chain just moves on - nothing breaks, it just falls back
further down the list (eventually to DuckDuckGo, which needs no config).
"""

import os
import re
import time
import logging
import requests
from typing import Optional, List, Dict
from bs4 import BeautifulSoup

try:
    from ddgs import DDGS  # current package name
except ImportError:
    from duckduckgo_search import DDGS  # fallback for older installs

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; NoahBot/1.0; +https://byncai.net)"
}

BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")
BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

SEARXNG_URL = os.getenv("SEARXNG_URL", "").rstrip("/")
SEARXNG_API_KEY = os.getenv("SEARXNG_API_KEY", "")


# --------------------
# Provider: Brave Search API
# --------------------
def _search_brave(query: str, max_results: int) -> List[Dict]:
    if not BRAVE_API_KEY:
        return []
    try:
        resp = requests.get(
            BRAVE_ENDPOINT,
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": BRAVE_API_KEY,
            },
            params={"q": query, "count": max_results},
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in data.get("web", {}).get("results", [])[:max_results]:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("description", ""),
            })
        if results:
            logger.info("Brave search succeeded for query=%r (%d results)", query, len(results))
        return results
    except Exception as e:
        logger.warning("Brave search failed for query=%r: %s", query, e)
        return []


# --------------------
# Provider: SearXNG
# --------------------
def _search_searxng(query: str, max_results: int) -> List[Dict]:
    if not SEARXNG_URL:
        return []
    try:
        headers = dict(HEADERS)
        if SEARXNG_API_KEY:
            headers["Authorization"] = f"Bearer {SEARXNG_API_KEY}"

        resp = requests.get(
            f"{SEARXNG_URL}/search",
            headers=headers,
            params={"q": query, "format": "json"},
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in data.get("results", [])[:max_results]:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", ""),
            })
        if results:
            logger.info("SearXNG search succeeded for query=%r (%d results)", query, len(results))
        return results
    except Exception as e:
        logger.warning("SearXNG search failed for query=%r: %s", query, e)
        return []


# --------------------
# Provider: DuckDuckGo (fallback, keyless)
# --------------------
def _search_duckduckgo(query: str, max_results: int) -> List[Dict]:
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href") or r.get("url", ""),
                    "snippet": r.get("body", ""),
                })
        if results:
            logger.info("DuckDuckGo search succeeded for query=%r (%d results)", query, len(results))
    except Exception as e:
        logger.warning("DuckDuckGo search failed for query=%r: %s", query, e)
    return results


# Order matters: Brave first (most reliable), then SearXNG, then DDG last.
_PROVIDERS = [
    ("brave", _search_brave),
    ("searxng", _search_searxng),
    ("duckduckgo", _search_duckduckgo),
]


def search_web(query: str, max_results: int = 5) -> List[Dict]:
    """
    Try each configured provider in order, returning the first one that
    yields results. Returns [] only if every provider fails/returns
    nothing - callers should treat that as "no web context available".
    """
    for name, fn in _PROVIDERS:
        results = fn(query, max_results)
        if results:
            return results
        logger.info("Provider %r returned nothing for query=%r, trying next.", name, query)
    logger.warning("All search providers failed for query=%r", query)
    return []


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
# Runs on essentially every non-KB message, so an identical or
# near-identical question arriving again within a short window (a user
# repeating themselves, a page refresh, two people asking the same thing)
# shouldn't re-hit any provider. Keeps things fast and avoids rate limits.
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


def resolve_site_url(name: str, max_results: int = 3) -> Optional[str]:
    """
    Lightweight lookup for "what's the actual URL for X" - no page
    fetching, just the first plausible link from a search result.

    Used by the browser agent (agent.py) so 'navigate' actions use a
    real, verified URL instead of the LLM guessing/hallucinating one.

    Cached under a "site:" prefix so repeated "open X" goals in the
    same session don't re-hit any search provider.
    """
    cache_key = f"site:{name.strip().lower()}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached or None  # cached "" means "looked it up, found nothing"

    url = None
    try:
        results = search_web(f"{name} official website", max_results=max_results)
        for r in results:
            candidate = r.get("url", "")
            if candidate.startswith("http"):
                url = candidate
                break
    except Exception as e:
        logger.exception("resolve_site_url failed for %r: %s", name, e)

    _cache_set(cache_key, url or "")
    return url


def build_web_context(query: str, max_results: int = 5, fetch_pages: bool = True,
                       max_pages_to_fetch: int = 3) -> str:
    """
    Search the web (Brave -> SearXNG -> DuckDuckGo) and assemble one
    context block, formatted so it's easy for the LLM to reference
    naturally:

        [1] Title — url
        snippet or fetched page text

        [2] Title — url
        ...

    Returns "" if every provider failed or returned nothing, so the
    caller can gracefully skip adding web context to the prompt.
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
