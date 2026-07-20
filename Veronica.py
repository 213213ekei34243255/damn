import json
from difflib import get_close_matches
from typing import Optional, List, Dict
from datetime import datetime
import os
from pathlib import Path
import google.generativeai as genai
import re

from openai import OpenAI

# Initialize DeepSeek client


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
import redis
import torch
import numpy as np
from sentence_transformers import util

# Global setup (preloaded model + embeddings)
from global_setup import model_embed, CHUNKS, CHUNK_EMBS

import google.generativeai as genai


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


def get_llama_response(user_question: str, session_id: str) -> str:

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

    retrieved_chunks = [
        CHUNKS[h["corpus_id"]]["text"]
        for h in hits
    ]

    context = "\n\n".join(
        c["text"] if isinstance(c, dict) else str(c)
        for c in retrieved_chunks
    )

    history = load_history(session_id, limit=8)

    messages = [
        {
            "role": "system",
            "content": """
           "You are Noah, Working in Jonah Browser you were made by CogniAI Studios , and your architecture is Rexy 1\n"
            "Answer all the questions asked by the user check Internet and then answer web related answers have modern american language like bruh and then being too human way"
        """
        }
    ]

    if context:
        messages.append({
            "role": "system",
            "content": f"Knowledge Base:\n{context}"
        })

    for m in history:
        messages.append({
            "role": m["role"],
            "content": m["text"]
        })

    messages.append({
        "role": "user",
        "content": user_question
    })

    payload = {
        "model": "local",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 512
    }

    try:

        r = requests.post(
            LLAMA_URL,
            json=payload,
            timeout=120
        )

        r.raise_for_status()

        result = r.json()

        return result["choices"][0]["message"]["content"].strip()

    except Exception as e:

        return f"Local model error: {e}"


# -------------------------------------------------
# Function to get Veronica's response based on KB
# -------------------------------------------------
from global_setup import DATA
def handle_stream_query(query, data):
    q = query.lower()

    mappings = data.get("mappings", {})
    fees = data.get("fees", {})

    results = []

    # ✅ 1. DIRECT STREAM MATCH (FIXES YOUR BUG)
    for stream in fees:
        if stream.lower() in q:
            return f"{stream} – ₹{fees[stream]}"

    # ✅ 2. CATEGORY MATCH (arts, science, commerce)
    for key, streams in mappings.items():
        if key in q:
            for s in streams:
                if s in fees:
                    results.append(f"{s} – ₹{fees[s]}")
            return "\n".join(results)

    return None
def get_veronica_response(user_question: str, knowledge_base: Dict, session_id: str) -> str:
    # quick utility commands
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

    # 🔥 NEW: Handle stream/fees BEFORE anything else
    if "fee" in user_question.lower() or "fees" in user_question.lower():
        stream_answer = handle_stream_query(user_question, DATA)
        if stream_answer:
            answer = stream_answer
            save_message(session_id, "user", user_question)
            save_message(session_id, "assistant", answer)
            return answer

    # Try FAQ/knowledge base first
    best_match = find_best_match(
        user_question,
        [q.get("question") for q in knowledge_base.get("questions", [])]
    )

    if best_match:
        answer = get_answer_for_question(best_match, knowledge_base) or "No answer found."
    else:
        # If no match is found in the knowledge base, ask Gemini to generate a response
         answer = get_llama_response(user_question, session_id)

    # Save this turn in Redis so future messages have context
    save_message(session_id, "user", user_question)
    save_message(session_id, "assistant", answer)

    return answer
# Main section for testing purposes
if __name__ == "__main__":
    knowledge_base = load_knowledge_base('knowledge_base.json')
    
    while True:
        user_question = input('You: ')
        if user_question.lower() == 'quit':
            break
        else:
            response = get_veronica_response(user_question, knowledge_base)
            print('Veronica:', response)
