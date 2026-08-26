"""
web_search.py
--------------
Multi-provider web-search augmentation for the Veronica / Noah chatbot.

Provider order (first success wins):
    1. Serper.dev            - official API wrapping real Google results,
                              authenticated (not scraped), so it isn't
                              subject to the CAPTCHA/IP-block problems
                              that hit scraper-based providers on cloud
                              hosts like Render.
    2. Brave Search API     - paid/free-tier key, kept as a second option
                              in case you get that account working later.
    3. SearXNG              - free, self-hosted (or public instance)
                              metasearch. Its scraped engines (Google,
                              Bing, DuckDuckGo, Brave, Startpage) tend to
                              get CAPTCHA'd from Render's IP ranges, so
                              in practice this mostly serves Wikipedia
                              results unless you're self-hosting SearXNG
                              somewhere with a cleaner IP.
    4. DuckDuckGo (ddgs)    - free, keyless, last-resort fallback (also
                              scraped, so also IP-block-prone).

Every provider is wrapped so a failure (timeout, bad key, rate limit,
CAPTCHA, empty result, exception of any kind) just falls through to the
next one. If all four fail, search_web() returns [] and callers already
treat that as "no web context available".

Install:
    pip install ddgs beautifulsoup4 requests

Environment variables:
    SERPER_API_KEY  - your serper.dev API key (get one at
                       https://serper.dev - free tier includes 2,500
                       searches/month, no credit card required)
    BRAVE_API_KEY   - your Brave Search API subscription token
                       (get one at https://brave.com/search/api/)
    SEARXNG_URL     - base URL of a SearXNG instance, e.g.
                       "https://searx.example.com" or your self-hosted
                       instance. If unset, SearXNG is skipped.
    SEARXNG_API_KEY - optional, only needed if your instance requires auth

If any of these are not set, that provider is silently skipped and the
chain just moves on - nothing breaks, it just falls back further down
the list. DuckDuckGo needs no config, so the chain always has a working
fallback even with zero env vars set.
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


# --------------------
# Search gating: decide whether a question actually needs a live web
# search, instead of searching on every single non-KB message. Searching
# for "Hey" or "what's photosynthesis" wastes an API call (and Serper's
# free tier is capped) for something the LLM already knows fine on its
# own or that isn't even a real question.
# --------------------

# Short greetings / chit-chat / acknowledgements - never worth a search,
# no matter what. Checked first, as a fast bailout.
_CHITCHAT_PATTERN = re.compile(
    r'^\s*(hi+|hey+|hello+|yo+|sup+|hola|howdy|good\s*(morning|afternoon|evening|night)|'
    r'thanks?( you)?|thank\s*you|ty|thx|ok(ay)?|cool|nice|great|awesome|lol+|haha+|'
    r'bye+|goodbye|see\s*ya|cya|what\'?s\s*up|how\s*are\s*you|how\'?s\s*it\s*going|'
    r'yes|yeah|yep|no|nope|sure|alright|got\s*it)\s*[!.?]*\s*$',
    re.IGNORECASE
)

# Explicit "go look this up for me" requests - the user is directly
# asking for a search, so always honor it regardless of topic. Two
# patterns: named phrases like "search for X", and a more general
# catch-all for any sentence that pairs an action verb (check/search/
# look/browse/google) with a target noun (web/internet/online) anywhere
# nearby - covers phrasings like "check the web", "you have to search
# online", "can you look online for this" that a fixed phrase list would
# otherwise miss.
_EXPLICIT_SEARCH_PATTERN = re.compile(
    r'\b(search (the web |online )?for|look up|check online|google (it|this|that)|'
    r'search the internet|browse (the web|online) for|find (me )?(info|information|news) (on|about))\b',
    re.IGNORECASE
)
_EXPLICIT_SEARCH_GENERIC_PATTERN = re.compile(
    r'\b(search|check|look|browse|google)\b[^.?!\n]{0,30}\b(web|internet|online)\b'
    r'|\b(web|internet|online)\b[^.?!\n]{0,30}\b(search|check|look|browse)\b',
    re.IGNORECASE
)

# Words/phrases that hint the user wants fresh, real-world, time-sensitive
# info - things a static LLM genuinely cannot know (scores, prices, who
# currently holds a role, breaking news, etc.). This is intentionally
# generous - a false positive just means we search a bit more than
# strictly necessary, which is harmless. A false negative means a stale
# answer, which is worse, so lean toward including borderline triggers.
WEB_SEARCH_TRIGGERS = [
    "latest", "today", "current", "currently", "news", "update", "updated",
    "right now", "this week", "this month", "this year", "recent", "recently",
    "price of", "stock price", "exchange rate", "score", "weather", "forecast",
    "who is the current", "who won", "who is winning", "release date",
    "when is", "when was", "when does", "what happened", "happening now",
    "election", "ceo of", "president of", "prime minister of",
    "is still", "is it still", "does still exist", "still around",
    "just announced", "breaking", "live", "schedule for", "upcoming",
]

# Generic-knowledge / conceptual-question phrasing - things the LLM
# should just answer from its own training, NOT search for, even though
# they're "questions." A trigger word above still overrides this (e.g.
# "explain the latest AI news" should still search).
_GENERIC_KNOWLEDGE_PATTERN = re.compile(
    r'^\s*(explain|what is|what are|how does|how do|why does|why do|why is|'
    r'define|describe|tell me about the theory|what does .* mean|'
    r'can you explain|help me understand|what\'?s the difference between)\b',
    re.IGNORECASE
)


def needs_web_search(question: str) -> bool:
    """
    Decide whether a question needs a live web search, rather than
    searching unconditionally on every message.

    Order of checks:
      1. Chit-chat / greetings -> never search.
      2. Explicit "search for X" style requests -> always search.
      3. Time-sensitive/current-event trigger words, or a recent year
         (2024-2029) -> search.
      4. Generic "explain/what is/how does" conceptual phrasing with no
         trigger word -> don't search (the LLM can answer this itself).
      5. Default: don't search. Most ordinary chat, opinions, small talk,
         math, general science, or explanations don't need it - only
         search when there's a real signal the LLM needs fresher info
         than its training data provides.
    """
    q = question.strip().lower()

    if not q:
        return False

    if _CHITCHAT_PATTERN.match(q):
        return False

    if _EXPLICIT_SEARCH_PATTERN.search(q):
        return True

    if any(trigger in q for trigger in WEB_SEARCH_TRIGGERS):
        return True

    # A "current-ish" year (2024-2029) is a strong signal something is
    # time-sensitive (an event, a release, a result) even without one of
    # the trigger phrases above.
    if re.search(r"\b(202[4-9])\b", q):
        return True

    if _GENERIC_KNOWLEDGE_PATTERN.match(q):
        return False

    return False

BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")
BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
SERPER_ENDPOINT = "https://google.serper.dev/search"

SEARXNG_URL = os.getenv("SEARXNG_URL", "").rstrip("/")
SEARXNG_API_KEY = os.getenv("SEARXNG_API_KEY", "")


# --------------------
# Provider: Serper.dev (wraps real Google results via an official API -
# authenticated request, not a scrape, so it isn't subject to the
# CAPTCHA/IP-block issues that hit SearXNG's scraped engines)
# --------------------
def _search_serper(query: str, max_results: int) -> List[Dict]:
    if not SERPER_API_KEY:
        return []
    try:
        resp = requests.post(
            SERPER_ENDPOINT,
            headers={
                "X-API-KEY": SERPER_API_KEY,
                "Content-Type": "application/json",
            },
            json={"q": query, "num": max_results},
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in data.get("organic", [])[:max_results]:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            })
        # Serper also returns a rich "answerBox" for direct-answer queries
        # (e.g. "who is the CEO of X") - surface it as a synthetic top
        # result when present, since it's often the single most useful
        # piece of context for exactly the kind of question that triggers
        # a web search in the first place.
        answer_box = data.get("answerBox")
        if answer_box:
            results.insert(0, {
                "title": answer_box.get("title", "Answer"),
                "url": answer_box.get("link", ""),
                "snippet": answer_box.get("snippet") or answer_box.get("answer", ""),
            })
        if results:
            logger.info("Serper search succeeded for query=%r (%d results)", query, len(results))
        return results
    except Exception as e:
        logger.warning("Serper search failed for query=%r: %s", query, e)
        return []


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


# Order matters: Serper first (official API, most reliable — not a
# scraper, so no CAPTCHA/IP-block risk), then Brave (in case you get
# that key working later), then SearXNG, then DuckDuckGo last as the
# final, keyless safety net.
_PROVIDERS = [
    ("serper", _search_serper),
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
    Search the web (Serper -> Brave -> SearXNG -> DuckDuckGo) and assemble
    one context block, formatted so it's easy for the LLM to reference
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
