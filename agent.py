import json
import os
import re
import logging
import threading
import requests
from typing import Dict, List, Any, Optional

# NEW: free, keyless web search (DuckDuckGo) support - used to resolve the
# real URL for "open X" / "go to X" goals instead of letting the LLM guess.
from web_search import resolve_site_url

# ================================
# Configuration
# ================================

LLAMA_URL = os.getenv(
    "LLAMA_URL",
    "http://127.0.0.1:8080/v1/chat/completions"
)

MODEL_NAME = os.getenv(
    "REXY_MODEL",
    "local"
)

# ---------------------------------------------------------------------
# FIX: the local LLM call used to have a single 120s timeout with a 3x
# retry loop around it (up to 360s before ever giving up), and no
# automatic fallback if that local server was unreachable or too slow to
# respond in time - which is exactly what "Read timed out (read
# timeout=120)" in the logs was. Two independent knobs now control this:
#
#   LLAMA_CONNECT_TIMEOUT_S - how long to wait for the TCP connection to
#     even establish. If nothing is listening on LLAMA_URL at all, this
#     fails in a few seconds instead of hanging for two minutes.
#
#   LLAMA_READ_TIMEOUT_S - how long to wait for a response once connected.
#     Kept generous by default since a CPU-bound local model can be slow,
#     but far below the old 120s so a single stuck request can't eat
#     minutes on its own.
#
# If the local model doesn't answer within these limits, call_llm()
# automatically falls back to Gemini (when GEMINI_API_KEY is configured)
# instead of failing the whole planning cycle.
# ---------------------------------------------------------------------
LLAMA_CONNECT_TIMEOUT_S = float(os.getenv("LLAMA_CONNECT_TIMEOUT_S", "5"))
LLAMA_READ_TIMEOUT_S = float(os.getenv("LLAMA_READ_TIMEOUT_S", "45"))

# NEW: dedicated, much shorter timeouts for the page-content classifier
# (see needs_page_content() below). It's a tiny yes/no JSON decision, not
# a full plan - if it can't get an answer almost immediately, defaulting
# to "no" and answering without page content is far better than making
# a normal chat message wait on a second slow LLM round trip.
CLASSIFIER_CONNECT_TIMEOUT_S = float(os.getenv("CLASSIFIER_CONNECT_TIMEOUT_S", "3"))
CLASSIFIER_READ_TIMEOUT_S = float(os.getenv("CLASSIFIER_READ_TIMEOUT_S", "8"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

_gemini_model = None
if GEMINI_API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        _gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    except Exception:
        logging.getLogger("NoahAgent").exception(
            "Gemini fallback configured but failed to initialize; "
            "falling back to local-only mode."
        )
        _gemini_model = None

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("NoahAgent")

if not GEMINI_API_KEY:
    logger.warning(
        "GEMINI_API_KEY not set - no fallback available if the local "
        "LLM at %s is slow or unreachable. Planning requests will fail "
        "fast instead of hanging, but won't have a second option.",
        LLAMA_URL,
    )


# ================================
# Supported Actions
# ================================

SUPPORTED_ACTIONS = {

    "navigate",

    "click",

    "type",

    "pressKey",

    "scroll",

    "wait",

    "observe",

    "extract",

    "reload",

    "goBack",

    "goForward",

    "download",

    "upload",

    "createTab",

    "closeTab",

    "switchTab",

    "complete"

}
ACTION_GO_BACK = "goBack"
ACTION_GO_FORWARD = "goForward"
ACTION_CREATE_TAB = "createTab"


# ================================
# Navigation target detection
# ================================

# Matches goals like "open amazon", "go to youtube", "navigate to the
# christ junior college website", "visit cogniai studios site".
# Captures the site NAME so we can look up its real URL via web search
# instead of letting the LLM guess/hallucinate a domain.
_NAV_PATTERN = re.compile(
    r'^\s*(?:open|go to|goto|navigate to|visit)\s+(?:the\s+)?(.+?)\s*(?:website|site|page|homepage)?\s*$',
    re.IGNORECASE
)


def extract_navigation_target(goal: str) -> Optional[str]:
    """
    If the goal is a simple "open/go to/visit X" request, return the
    site name X. Returns None for anything else (multi-step goals,
    goals that already contain a URL, clicking/typing instructions,
    etc.) - those are left to the model as before.
    """
    if not goal:
        return None

    match = _NAV_PATTERN.match(goal.strip())
    if not match:
        return None

    target = match.group(1).strip()

    if not target:
        return None

    # Already a URL - nothing to resolve, let sanitize_action handle it.
    if target.lower().startswith(("http://", "https://", "www.")):
        return None

    return target


# ================================
# On-demand page-content decision (JSON/tool-calling)
# ================================

# NEW: fast local pre-filter for the obvious case. If the message
# plainly asks about "this page" / "summarize" / etc., there's no need
# to spend an LLM call just to confirm what's already certain - this
# keeps the common, explicit case (the Summarize button, or someone
# typing "summarize this") just as fast as before, with zero added
# round trips, while still being a real, deterministic decision made
# before any page content is touched.
_PAGE_CONTENT_HINTS = re.compile(
    r'\b(this page|this site|this article|this website|current page|'
    r'summariz|summary|tl;?dr|what does this say|what is this about)\b',
    re.IGNORECASE
)


def _classify_needs_page_content_via_llm(message: str) -> Optional[bool]:
    """
    The actual JSON/tool-calling decision: ask the model itself whether
    it needs the page's text to answer well. Returns True/False, or None
    if the classifier call failed for any reason (caller should then
    default to False rather than block the chat on a broken classifier).
    """
    prompt = (
        "A browser assistant received this message from a user who has a "
        "webpage open:\n\n"
        f"\"{message}\"\n\n"
        "Decide whether answering this well requires reading the actual "
        "text/content of the currently open webpage (e.g. summarizing it, "
        "answering a question about what's on the page, extracting "
        "specific information from it) versus a general question, greeting, "
        "or something answerable without seeing the page at all.\n\n"
        "Respond with ONLY this JSON object and nothing else:\n"
        '{"needs_page_content": true}\n'
        "or\n"
        '{"needs_page_content": false}'
    )
    try:
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 20,
        }
        response = requests.post(
            LLAMA_URL,
            json=payload,
            timeout=(CLASSIFIER_CONNECT_TIMEOUT_S, CLASSIFIER_READ_TIMEOUT_S),
        )
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"].strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return None
        parsed = json.loads(text[start:end + 1])
        value = parsed.get("needs_page_content")
        return bool(value) if isinstance(value, bool) else None
    except Exception as e:
        logger.warning(f"needs_page_content classifier failed: {e}")
        return None


def needs_page_content(message: str) -> bool:
    """
    The single entry point app.py's chat branch calls before deciding
    whether to include page content in the prompt at all. This is what
    "stop sending website contents with every chat message" and "expose
    summarize through JSON/tool calling" both cash out to concretely:
    page content is never attached by default - it's only fetched and
    used once something (the fast local hint match, or the LLM
    classifier for ambiguous phrasing) has actually decided it's needed
    for THIS message.

    Fails safe: any classifier error defaults to False (don't require
    page content) rather than blocking or erroring the chat.
    """
    if not message:
        return False

    if _PAGE_CONTENT_HINTS.search(message):
        return True

    result = _classify_needs_page_content_via_llm(message)
    return result if result is not None else False


# ================================
# Planner Prompt
# ================================

SYSTEM_PROMPT = """
You are Noah AI.

Codename: Rexy.

You are NOT a chatbot.

You are an autonomous browser agent.

Your responsibility is to convert user goals into executable plans.

You NEVER explain.

You NEVER answer conversationally.

You ONLY produce JSON.
CRITICAL:

If the requested website is already open,

DO NOT navigate again.

Return

{
    "complete": true,
    "reason":"Already on requested page.",
    "actions":[]
}

Examples

Goal:
Open Amazon

Current URL:
https://www.amazon.com

Output

{
 "complete":true,
 "reason":"Already on Amazon",
 "actions":[]
}

Every response MUST follow this schema:

{
    "complete": false,
    "reason": "",
    "actions": [

    ]
}
Example of a CORRECT response — navigating to a known website:

{
    "complete": false,
    "reason": "Navigating to Amazon",
    "actions": [
        { "type": "navigate", "url": "https://www.amazon.com" }
    ]
}

Example of a CORRECT response — typing into a field:

{
    "complete": false,
    "reason": "Typing into the search box",
    "actions": [
        { "type": "type", "selector": "#urlBar", "text": "amazon.com" }
    ]
}

The "type" action REQUIRES exactly these two fields: "selector" and "text".
Never use "value" or "content" — the field is always called "text".

Example of an INCORRECT response (never do this — actions must be
objects with a "type" field, never bare strings):

{
    "actions": ["navigate", "type"]
}

CRITICAL RULES:

1. If the goal is to open a known website (e.g. "Open Amazon", "Go to
   YouTube"), ALWAYS use a single "navigate" action with the full URL.
   Do NOT use "type" or "click" to open a website.

1b. If a "VERIFIED WEBSITE URL" is provided below in the mission, that
    URL has already been confirmed via live web search — it is the
    correct destination. Use it exactly as given in your "navigate"
    action. Do NOT substitute a different URL you think might be right.

2. NEVER target browser chrome elements — window controls, minimize
   buttons, maximize buttons, close buttons, tab bars, or menu icons.
   These control the browser application itself, not the webpage, and
   interacting with them will break the user's session. Only interact
   with elements that are part of the actual page content (inputs,
   links, buttons within the loaded page).

3. If the current page is a blank/home page and the goal names a
   specific website, prefer "navigate" over trying to find a search
   box on the current page.

Supported action types are:

navigate
click
type
pressKey
scroll
wait
observe
extract
reload
goBack
goForward
download
upload
createTab
closeTab
switchTab
complete

Every action should contain only the parameters required for that action.

Never return markdown.

Never return code fences.

Never return plain text.

Only valid JSON.
"""


# ================================
# Noah Planner
# ================================

class AgentPlanner:

    def __init__(self):

        self.session_memory = {}
        # NEW: app.py now runs Flask with threaded=True so a slow request
        # (an agent plan, a classifier call, a Veronica response) never
        # blocks every other in-flight request - including other agent
        # sessions hitting this same shared dict concurrently. This lock
        # keeps remember()/recall() race-free under that concurrency.
        self._memory_lock = threading.Lock()

        logger.info("🧠 Noah Agent Planner Initialized")

    def remember(self,
                 session_id: str,
                 memory: Dict):

        with self._memory_lock:
            self.session_memory[session_id] = memory

    def recall(self,
               session_id: str):

        with self._memory_lock:
            return self.session_memory.get(session_id, {})

    def build_prompt(
            self,
            goal: str,
            observation: Dict,
            memory: Dict,
            resolved_url: Optional[str] = None
    ):

        """
        Build the reasoning prompt for Noah.
        """

        observation_json = json.dumps(
            self.summarize_observation(observation),
            indent=2,
            ensure_ascii=False
        )

        # FIX: this used to be `json.dumps(memory, ...)` with NO trimming
        # at all — the full raw memory blob (cookies, full page snapshot,
        # unbounded recentActions, etc.) got serialized straight into the
        # prompt text sent to the LLM on every single planning cycle. That
        # inflates the input size the model has to process, which costs
        # real time on a CPU-bound local model. summarize_memory() keeps
        # only what's actually useful for deciding the next action.
        memory_json = json.dumps(
            self.summarize_memory(memory),
            indent=2,
            ensure_ascii=False
        )

        # NEW: if we resolved a real URL via web search for this goal,
        # tell the model exactly what to use instead of letting it guess.
        resolved_block = ""
        if resolved_url:
            resolved_block = f"""
    =========================
    VERIFIED WEBSITE URL
    =========================

    Live web search confirms the correct destination for this goal is:

    {resolved_url}

    Use this exact URL in your "navigate" action. Do not invent or modify it.
    """

        user_prompt = f"""
    =========================
    MISSION
    =========================

    Your goal:

    {goal}
    {resolved_block}

    =========================
    CURRENT BROWSER STATE
    =========================

    {observation_json}


    =========================
    MEMORY
    =========================

    {memory_json}


    =========================
    INSTRUCTIONS
    =========================

    Think like an autonomous AI agent.

    Observe the browser.

    Reason about the next step.

    If the goal has already been completed,
    return:

    {{
        "complete": true,
        "reason": "...",
        "actions":[]
    }}

    Otherwise return ONLY the next actions.

    Never return explanations.

    Never return markdown.

    Never return code.

    Only valid JSON.

    Keep plans short.

    Do NOT generate more than 5 actions.

    """

        return [

            {

                "role": "system",

                "content": SYSTEM_PROMPT

            },

            {

                "role": "user",

                "content": user_prompt

            }

        ]

    def summarize_observation(
            self,
            observation: Dict
    ):

        """
        Reduce unnecessary HTML before sending
        to the LLM. buttons/inputs/links/text live under
        observation["page"], not at the top level.
        """

        page = observation.get("page", {}) or {}
        browser = observation.get("browser", {}) or {}

        def _slim_button(b):
            return {"text": (b.get("text") or "")[:80], "selector": b.get("selector", "")}

        def _slim_input(i):
            return {
                "type": i.get("type", ""),
                "placeholder": (i.get("placeholder") or "")[:60],
                "name": (i.get("name") or "")[:60],
                "selector": i.get("selector", ""),
            }

        def _slim_link(l):
            return {"text": (l.get("text") or "")[:80], "href": l.get("href", ""), "selector": l.get("selector", "")}

        buttons = [b for b in page.get("buttons", []) if b.get("visible") is not False]
        inputs = page.get("inputs", [])
        links = [l for l in page.get("links", []) if l.get("visible") is not False]

        summary = {
            "title": browser.get("title") or observation.get("title"),
            "url": browser.get("url") or observation.get("url"),
            "buttons": [_slim_button(b) for b in buttons[:20]],
            "inputs": [_slim_input(i) for i in inputs[:20]],
            "links": [_slim_link(l) for l in links[:20]],
            "text": (page.get("text") or observation.get("text") or "")[:4000],
        }

        return summary

    def summarize_memory(
            self,
            memory: Dict
    ):
        """
        NEW: mirrors summarize_observation() but for the memory blob.
        Keeps only what actually helps the model decide the next step -
        the goal, current task progress, and a short tail of recent
        actions - and drops everything else (cookies, full page/browser
        snapshots already covered by summarize_observation, unbounded
        history, etc.) that was previously dumped in raw.
        """
        if not isinstance(memory, dict):
            return {}

        recent_actions = memory.get("recentActions") or []
        slim_actions = [
            {
                "action": a.get("action"),
                "args": a.get("args"),
                "success": bool((a.get("result") or {}).get("success")),
                "goal": a.get("goal"),
            }
            for a in recent_actions[-5:]
            if isinstance(a, dict)
        ]

        task = memory.get("task") or {}

        return {
            "goal": (memory.get("goal") or {}).get("text"),
            "task": {
                "currentTask": task.get("currentTask"),
                "completed": (task.get("completed") or [])[-5:],
                "pending": (task.get("pending") or [])[:5],
            },
            "recentActions": slim_actions,
            "lastResponse": (memory.get("llmContext") or {}).get("lastResponse", ""),
        }

    def merge_memory(
            self,
            session_id,
            runtime_memory
    ):

        saved = self.recall(session_id)

        merged = {}

        merged.update(saved)

        merged.update(runtime_memory)

        return merged

    def _call_local_llm(self, messages: List[Dict]) -> str:
        """
        Call the local LLM at LLAMA_URL with fast-fail timeouts. Raises
        on any failure (connection error, timeout, non-2xx, bad body) -
        callers decide what to do next (retry, fall back, or give up).
        """
        payload = {"model": MODEL_NAME, "messages": messages, "temperature": 0.15, "max_tokens": 1024}
        logger.info("🧠 Sending planning request to local LLM (%s)...", LLAMA_URL)
        response = requests.post(
            LLAMA_URL,
            json=payload,
            timeout=(LLAMA_CONNECT_TIMEOUT_S, LLAMA_READ_TIMEOUT_S),
        )
        if not response.ok:
            logger.warning(
                "Local LLM request failed [%s]: %s",
                response.status_code, response.text[:2000],
            )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()

    def _call_gemini(self, messages: List[Dict]) -> str:
        """
        Fallback path when the local LLM is unreachable/too slow. Only
        used if GEMINI_API_KEY is configured and google.generativeai
        initialized successfully at import time.
        """
        if not _gemini_model:
            raise RuntimeError("Gemini fallback not configured (GEMINI_API_KEY missing).")

        logger.info("🧠 Falling back to Gemini (%s)...", GEMINI_MODEL_NAME)
        # Gemini doesn't use the OpenAI-style role list directly; flatten
        # the system + user messages into a single prompt instead.
        combined = "\n\n".join(m["content"] for m in messages)
        response = _gemini_model.generate_content(
            combined,
            generation_config={"temperature": 0.15, "max_output_tokens": 1024},
        )
        return (response.text or "").strip()

    def call_llm(self, messages: List[Dict]):
        """
        Try the local LLM first (fast-fail on connect/read timeout), and
        automatically fall back to Gemini if it fails and a fallback is
        configured. This is what actually fixes "plan timed out": instead
        of hanging for up to 120s per attempt with nothing to fall back
        on, an unreachable/slow local server now fails within
        LLAMA_CONNECT_TIMEOUT_S + LLAMA_READ_TIMEOUT_S seconds and hands
        off immediately.
        """
        try:
            return self._call_local_llm(messages)
        except (requests.ConnectionError, requests.Timeout) as e:
            logger.warning(f"Local LLM unreachable/slow ({e}); trying fallback.")
        except requests.HTTPError as e:
            logger.warning(f"Local LLM returned an error ({e}); trying fallback.")

        return self._call_gemini(messages)

    def extract_json(
            self,
            text: str
    ):

        start = text.find("{")

        end = text.rfind("}")

        if start == -1 or end == -1:

            raise ValueError("No JSON found.")

        return json.loads(text[start:end + 1])

    def validate(
            self,
            plan: Dict
    ):

        if not isinstance(plan, dict):
            raise ValueError("Plan must be an object.")

        if "actions" not in plan:
            plan["actions"] = []

        if "complete" not in plan:
            plan["complete"] = False

        valid_actions = []

        for action in plan["actions"]:

            # NEW: coerce bare strings like "navigate" into {"type": "navigate"}
            if isinstance(action, str):
                action = {"type": action}

            if not isinstance(action, dict):
                continue

            action_type = action.get("type")

            if action_type not in SUPPORTED_ACTIONS:
                logger.warning(f"Unsupported action: {action_type}")
                continue

            valid_actions.append(self.sanitize_action(action))

        plan["actions"] = valid_actions

        return plan

    def generate_plan(
            self,
            messages
    ):
        """
        FIX: previously retried up to 3 times with NO distinction between
        "the model's JSON was malformed" (worth retrying - the model may
        do better on a second try) and "the LLM call itself failed"
        (retrying the exact same network failure 3 times just triples
        the wait for an identical outcome, since call_llm() already
        tried both the local model AND the Gemini fallback once each).
        Now: an LLM-call failure (both primary and fallback exhausted)
        gives up immediately; only bad-JSON/validation failures get
        retried, and only once (max_parse_retries=2 total attempts).
        """
        max_parse_retries = 2
        parse_attempts = 0
        last_error = None

        while parse_attempts < max_parse_retries:
            try:
                text = self.call_llm(messages)
            except Exception as e:
                # Both the local model and the Gemini fallback (if any)
                # failed. No amount of retrying will fix an unreachable
                # backend right now - fail fast instead of hanging.
                logger.error(f"call_llm failed on both primary and fallback: {e}")
                return {
                    "complete": False,
                    "reason": "Planner backend unavailable.",
                    "actions": [
                        {"type": "observe"}
                    ]
                }

            parse_attempts += 1
            try:
                plan = self.extract_json(text)
                raw_action_count = len(plan.get("actions", []))

                validated = self.validate(plan)

                # NEW: if the model gave us actions but validation stripped
                # all of them (malformed schema), treat this as a failed
                # attempt and retry rather than silently returning empty.
                if raw_action_count > 0 and len(validated["actions"]) == 0 and not validated.get("complete"):
                    logger.warning(
                        f"Model returned {raw_action_count} action(s) but all failed validation; retrying."
                    )
                    last_error = ValueError("All actions failed schema validation")
                    continue

                logger.info(json.dumps(validated, indent=4))
                return validated

            except Exception as e:
                logger.warning(e)
                last_error = e

        logger.error(last_error)

        return {
            "complete": False,
            "reason": "Planner failed.",
            "actions": [
                {"type": "observe"}
            ]
        }

    def sanitize_action(
            self,
            action
    ):

        # Normalize common field-name mistakes the model makes.
        if action["type"] == "type":
            if "text" not in action and "value" in action:
                action["text"] = action.pop("value")
            if "text" not in action and "content" in action:
                action["text"] = action.pop("content")

        if action["type"] == "navigate":

            url = action.get("url", "") or action.get("value", "") or action.get("href", "")

            if url and not url.startswith("http"):
                url = "https://" + url

            action["url"] = url

        if action["type"] == "click" and "selector" not in action and "target" in action:
            action["selector"] = action.pop("target")

        return action

    def apply_resolved_navigation(
            self,
            plan: Dict,
            resolved_url: Optional[str]
    ) -> Dict:
        """
        NEW: if we resolved a verified URL via web search for this goal,
        force every 'navigate' action in the plan to use it. This doesn't
        rely on the LLM obeying the prompt instruction - it's a hard
        guarantee that the agent never navigates to a hallucinated domain
        when we already know the real one.
        """
        if not resolved_url:
            return plan

        for action in plan.get("actions", []):
            if action.get("type") == "navigate":
                if action.get("url") != resolved_url:
                    logger.info(
                        f"Overriding navigate URL with web-search-verified URL: "
                        f"{action.get('url')!r} -> {resolved_url!r}"
                    )
                action["url"] = resolved_url

        return plan

    def plan(
            self,
            goal: str,
            observation: Dict,
            memory: Dict,
            session_id: str = "default"
    ):

        logger.info("====================================")
        logger.info("🧠 NEW PLANNING CYCLE")
        logger.info("====================================")

        logger.info(f"Goal: {goal}")

        # Merge runtime memory with saved session memory
        merged_memory = self.merge_memory(
            session_id,
            memory
        )

        # Save latest memory
        self.remember(
            session_id,
            merged_memory
        )

        # NEW: if this looks like "open X" / "go to X", resolve the real
        # URL via web search before we even ask the LLM to plan.
        resolved_url = None
        nav_target = extract_navigation_target(goal)
        if nav_target:
            try:
                resolved_url = resolve_site_url(nav_target)
                if resolved_url:
                    logger.info(f"Resolved '{nav_target}' -> {resolved_url}")
                else:
                    logger.info(f"Could not resolve a URL for '{nav_target}' via web search.")
            except Exception as e:
                logger.warning(f"resolve_site_url failed for {nav_target!r}: {e}")
                resolved_url = None

        # Build LLM prompt
        messages = self.build_prompt(
            goal,
            observation,
            merged_memory,
            resolved_url=resolved_url
        )

        # Ask the model
        plan = self.generate_plan(messages)

        # Guarantee the navigate action uses the verified URL, if we have one
        plan = self.apply_resolved_navigation(plan, resolved_url)

        logger.info("Planning complete.")

        return plan



planner = AgentPlanner()


def get_agent_plan(
        goal: str,
        observation: Dict,
        memory: Dict,
        session_id: str = "default"
):

    return planner.plan(

        goal=goal,

        observation=observation,

        memory=memory,

        session_id=session_id

    )
