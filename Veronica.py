import json
from difflib import get_close_matches
from typing import Optional, List, Dict
from datetime import datetime
import os
from pathlib import Path
import google.generativeai as genai
import re

from openai import OpenAI

# NEW: free, keyless web search (DuckDuckGo) support
from web_search import build_web_context, needs_web_search


# Function to load the knowledge base from a JSON file
def load_knowledge_base(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except json.JSONDecodeError:
        print(f"Error: Unable to parse JSON file '{file_path}'.")
    return {"questions": []}

# Function to save updated data to the knowledge base
def save_knowledge_base(file_path: str, data: dict) -> None:
    if data.get("questions"):
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=2)

# Function to find the best match for the user question from knowledge base
def find_best_match(user_question: str, questions: List[str]) -> Optional[str]:
    matches = get_close_matches(user_question, [q for q in questions if q is not None], n=1, cutoff=0.6)
    return matches[0] if matches else None

# Function to retrieve answer from the knowledge base
def get_answer_for_question(question: str, knowledge_base: Dict) -> Optional[str]:
    for q in knowledge_base.get("questions", []):
        if q.get("question") == question:
            return q.get("answer")
    return None

import os
import json
import logging
import redis
import torch
import numpy as np
from sentence_transformers import util

# Global setup (preloaded model + embeddings)
from global_setup import model_embed, CHUNKS, CHUNK_EMBS

import google.generativeai as genai

logger = logging.getLogger("Veronica")

# Set VERONICA_DEBUG_PROMPTS=1 in your environment to log the exact
# messages array sent to the LLM for every turn. Useful for diagnosing
# cases where the model seems to ignore web search / RAG context, OR
# where the model is refusing to answer altogether (as opposed to a
# plumbing/search failure) - you can see precisely what it was given
# right before it declined.
DEBUG_PROMPTS = os.getenv("VERONICA_DEBUG_PROMPTS", "0") == "1"

# NEW: strip raw URLs out of web_context before it ever reaches the
# model's prompt. Two reasons this matters:
#   1. The client's ghost-search "Source links:" block isn't just article
#      URLs - it includes Google's own chrome links: accounts.google.com
#      ServiceLogin redirects with an encoded `continue=` param, and
#      googleapp://lens deep links via iga.google.com. Those resemble
#      the kind of tracking/phishing-style redirect URLs some
#      safety-tuned local models are trained to flag, and could be
#      contributing to outright refusals on unrelated questions.
#   2. Even setting that aside, raw long URLs add prompt-token noise for
#      zero informational value - the model only needs the headline/
#      snippet text, never the literal link.
# Note: this is one candidate fix for model refusals, not a guaranteed
# one - a refusal can also be topic-driven (e.g. a mass-casualty
# disaster question), which stripping URLs won't change. Use the
# refusal-warning log added below to check whether refusals stop after
# this change; if they don't, the cause is more likely topical.
def _strip_urls_from_web_context(text: str) -> str:
    if not text:
        return text
    # Cut off the "Source links:" section entirely - pure link dump,
    # no informational value on its own.
    text = re.split(r'\n\s*Source links:\s*\n', text, maxsplit=1)[0]
    # Strip any remaining raw URLs elsewhere in the text (e.g. the
    # "[n] Title — url" format used by web_search.py's build_web_context
    # fallback path), leaving just the title/snippet content.
    text = re.sub(r'https?://\S+', '', text)
    # Clean up dangling separators/whitespace left behind (e.g.
    # "Title — " with nothing after the dash) and collapse blank lines.
    text = re.sub(r'—\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'[ \t]+\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# Minimum semantic-search similarity score (0-1) for a knowledge-base
# chunk to be included in the prompt. Below this, a chunk is almost
# certainly irrelevant to the question (e.g. college document chunks
# surfacing for a totally unrelated question like a sports score) and
# just adds noise that can distract the model from the actual answer -
# or from fresh web search results sitting right next to it. Tune this
# if you find relevant chunks getting filtered out, or irrelevant ones
# still getting through.
RAG_SCORE_THRESHOLD = float(os.getenv("VERONICA_RAG_SCORE_THRESHOLD", "0.35"))


# --------------------
# Redis Configuration
# --------------------
REDIS_URL = os.getenv(
    "REDIS_URL",
    "rediss://default:AYJVAAIncDJkYmE1M2FiNWMwNTk0OTI2OWVhODVmMDcxNzU1YjEwYXAyMzMzNjU@related-ox-33365.upstash.io:6379"
)

redis_client = redis.Redis.from_url(
    REDIS_URL,
    decode_responses=True
)


# --------------------
# Redis Helpers
# --------------------
def _chat_key(session_id: str) -> str:
    return f"veronica:chat:{session_id}"


def save_message(session_id: str, role: str, text: str) -> None:
    """
    Store a single message in Redis as a JSON string.
    role: 'user' or 'assistant'
    """
    key = _chat_key(session_id)

    redis_client.rpush(
        key,
        json.dumps({"role": role, "text": text})
    )

    # Auto-expire after 7 days
    redis_client.expire(key, 7 * 24 * 60 * 60)


def load_history(session_id: str, limit: int = 5):
    """
    Load the last `limit` messages from Redis.
    Returns: List[Dict] -> {role, text}
    """
    key = _chat_key(session_id)
    raw_messages = redis_client.lrange(key, -limit, -1)

    return [json.loads(msg) for msg in raw_messages]


# --------------------
# Main Gemini + RAG
# --------------------

import requests

LLAMA_URL = "http://127.0.0.1:8080/v1/chat/completions"


def get_llama_response(
    user_question: str,
    session_id: str,
    web_context: str = "",
    page_content: str = ""
) -> str:

    # Encode query
    user_emb = model_embed.encode(
        user_question,
        convert_to_tensor=True
    )

    chunk_embs_tensor = torch.tensor(CHUNK_EMBS)

    hits = util.semantic_search(
        user_emb,
        chunk_embs_tensor,
        top_k=20
    )[0]

    relevant_hits = [h for h in hits if h.get("score", 0) >= RAG_SCORE_THRESHOLD]

    retrieved_chunks = [
        CHUNKS[h["corpus_id"]]["text"]
        for h in relevant_hits
    ]

    context = "\n\n".join(
        c["text"] if isinstance(c, dict) else str(c)
        for c in retrieved_chunks
    )

    history = load_history(session_id, limit=8)

    system_sections = ["""
           "You are Noah, Working in Jonah Browser you were made by CogniAI Studios , and your architecture is Rexy 1\n"
            "Answer all the questions asked by the user check Internet and then answer web related answers have modern american language like bruh and then being too human way\n"
            "When you are given a 'Web Search Results' block below, treat it as freshly retrieved, up-to-date information - use it to answer, "
            "summarize it in your own words, and mention it naturally (e.g. 'from what I just found online...'). "
            "If the web results don't actually answer the question, say so instead of making something up. "
            "Never dump raw links or citation numbers like [1] into your reply - just talk about what you found.\n"
            "IMPORTANT: A 'Web Search Results' block, when present, is ALWAYS more current than anything you "
            "already know or said earlier in this conversation. Your own training data has a knowledge cutoff and "
            "can be out of date - if a Web Search Results block conflicts with your training knowledge OR with "
            "something said earlier in this chat history, the Web Search Results block wins. Never repeat or default "
            "back to an older answer from earlier in the conversation once fresher Web Search Results are provided.\n"
            "When a 'Current Web Page' block below is present, that is the actual text of the page the user has "
            "open right now in their browser. Use it as the primary source for summarizing the page or answering "
            "any question about 'this page'/'this article'/'this site'. It reflects exactly what the user is "
            "looking at - trust it over your own general knowledge about the topic if the two ever disagree.\n"
            "NEW: The Web Search Results are ordinary public search-engine results - the same thing anyone would "
            "see typing the question into Google themselves. This includes routine entertainment/news topics like "
            "movie casting, actors playing roles, release dates, and celebrity news reported by mainstream outlets "
            "and fan communities - this is public, widely-reported information, not private, sensitive, or harmful "
            "content, so answer these questions directly and factually using the Web Search Results provided. If "
            "a result describes something as a rumor, leak, or unconfirmed report, say so plainly as part of your "
            "answer (e.g. 'it's being reported/rumored that...') rather than declining to discuss it at all. "
            "Do not refuse to answer, and do not add disclaimers about being unable to discuss real people, when "
            "the question is this kind of everyday, publicly-reported information.
        """]

    if context:
        system_sections.append(
            f"Knowledge Base (only use this if it's actually relevant to the question below):\n{context}"
        )

    if web_context:
        web_context = _strip_urls_from_web_context(web_context)

    if web_context:
        system_sections.append(
            "Web Search Results (live, fetched just now - this is more "
            "current than your training data and more current than "
            "anything said earlier in this conversation):\n"
            f"{web_context}"
        )

    if page_content:
        system_sections.append(
            f"Current Web Page (the page the user has open right now):\n{page_content}"
        )

    messages = [
        {
            "role": "system",
            "content": "\n\n".join(system_sections)
        }
    ]

    for m in history:
        messages.append({
            "role": m["role"],
            "content": m["text"]
        })

    messages.append({
        "role": "user",
        "content": user_question
    })

    if DEBUG_PROMPTS:
        logger.info(
            "Final prompt for session=%s question=%r:\n%s",
            session_id, user_question,
            json.dumps(messages, indent=2, ensure_ascii=False)
        )

    def _post(msgs):
        return requests.post(
            LLAMA_URL,
            json={"model": "local", "messages": msgs, "temperature": 0.7, "max_tokens": 512},
            timeout=120
        )

    try:

        r = _post(messages)
        r.raise_for_status()
        result = r.json()
        reply = result["choices"][0]["message"]["content"].strip()

        # NEW: log (not silently swallow) apparent model refusals, so
        # they show up clearly in server logs next to the exact prompt
        # (when VERONICA_DEBUG_PROMPTS=1) instead of just looking like
        # an ordinary answer. Doesn't change behavior - purely visibility,
        # so you can tell "the model declined" apart from "the model
        # gave a real but wrong/unhelpful answer" at a glance in logs.
        _REFUSAL_MARKERS = (
            "i'm sorry, but i can't", "i am sorry but i cant",
            "i can't assist with that", "i cannot assist with that",
            "i'm not able to help with that", "as an ai",
        )
        if any(m in reply.lower() for m in _REFUSAL_MARKERS):
            logger.warning(
                "Local model appears to have REFUSED session=%s question=%r reply=%r",
                session_id, user_question, reply
            )

        return reply

    except requests.HTTPError as e:
        body = ""
        try:
            body = e.response.text[:500] if e.response is not None else ""
        except Exception:
            pass
        logger.warning(
            "Local LLM rejected request with a system message [%s]: %s - retrying without a system role.",
            e.response.status_code if e.response is not None else "?",
            body,
        )

        try:
            no_system_messages = []
            for i, m in enumerate(messages):
                if m["role"] == "system":
                    if i + 1 < len(messages):
                        continue
                    no_system_messages.append({"role": "user", "content": m["content"]})
                else:
                    no_system_messages.append(m)
            system_text = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
            if system_text and no_system_messages and no_system_messages[0]["role"] != "system":
                no_system_messages[0] = {
                    "role": no_system_messages[0]["role"],
                    "content": f"{system_text}\n\n{no_system_messages[0]['content']}"
                }

            r2 = _post(no_system_messages)
            r2.raise_for_status()
            result2 = r2.json()
            return result2["choices"][0]["message"]["content"].strip()

        except Exception as retry_error:
            retry_body = ""
            try:
                retry_body = retry_error.response.text[:500] if getattr(retry_error, "response", None) is not None else ""
            except Exception:
                pass
            logger.error(
                "Retry without system role also failed: %s | server said: %s",
                retry_error, retry_body,
            )
            return f"Local model error: {e} | server said: {body or retry_body}"

    except Exception as e:

        return f"Local model error: {e}"


from global_setup import DATA
def handle_stream_query(query, data):
    q = query.lower()

    mappings = data.get("mappings", {})
    fees = data.get("fees", {})

    results = []

    for stream in fees:
        if stream.lower() in q:
            return f"{stream} – ₹{fees[stream]}"

    for key, streams in mappings.items():
        if key in q:
            for s in streams:
                if s in fees:
                    results.append(f"{s} – ₹{fees[s]}")
            return "\n".join(results)

    return None


def get_veronica_response(
    user_question: str,
    knowledge_base: Dict,
    session_id: str,
    page_content: str = "",
    web_content: str = ""
) -> str:
    if user_question.lower() == 'date':
        answer = f"Today's date is {datetime.now().strftime('%Y-%m-%d')}"
        save_message(session_id, "user", user_question)
        save_message(session_id, "assistant", answer)
        return answer

    if user_question.lower() == 'time':
        answer = f"The current time is {datetime.now().strftime('%H:%M:%S')}"
        save_message(session_id, "user", user_question)
        save_message(session_id, "assistant", answer)
        return answer

    if "fee" in user_question.lower() or "fees" in user_question.lower():
        stream_answer = handle_stream_query(user_question, DATA)
        if stream_answer:
            answer = stream_answer
            save_message(session_id, "user", user_question)
            save_message(session_id, "assistant", answer)
            return answer

    best_match = find_best_match(
        user_question,
        [q.get("question") for q in knowledge_base.get("questions", [])]
    )

    if best_match:
        answer = get_answer_for_question(best_match, knowledge_base) or "No answer found."
    else:
        web_context = ""
        if web_content:
            web_context = web_content
        elif not page_content and needs_web_search(user_question):
            try:
                web_context = build_web_context(user_question)
            except Exception:
                web_context = ""

        answer = get_llama_response(
            user_question,
            session_id,
            web_context=web_context,
            page_content=page_content
        )

    save_message(session_id, "user", user_question)
    save_message(session_id, "assistant", answer)

    return answer

if __name__ == "__main__":
    knowledge_base = load_knowledge_base('knowledge_base.json')

    while True:
        user_question = input('You: ')
        if user_question.lower() == 'quit':
            break
        else:
            response = get_veronica_response(user_question, knowledge_base)
            print('Veronica:', response)
