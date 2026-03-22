import json
from difflib import get_close_matches
from typing import Optional, List, Dict
from datetime import datetime
import os
from pathlib import Path
import google.generativeai as genai
import re

# Configure the Gemini model with the API key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise EnvironmentError(
        "GEMINI_API_KEY not found in environment variables"
    )

genai.configure(api_key=api_key)

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


def load_history(session_id: str, limit: int = 3):
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
def get_gemini_response(user_question: str, session_id: str) -> str:
    # 1) Encode user query
    user_emb = model_embed.encode(
        user_question,
        convert_to_tensor=True
    )

    # Convert stored numpy embeddings → torch tensor
    chunk_embs_tensor = torch.tensor(CHUNK_EMBS)

    # Semantic search
    hits = util.semantic_search(
        user_emb,
        chunk_embs_tensor,
        top_k=3
    )[0]

    retrieved_chunks = [
        CHUNKS[h["corpus_id"]] for h in hits
    ]

    MAX_CONTEXT_CHARS = 1200
    context = "\n".join(retrieved_chunks).strip()[:MAX_CONTEXT_CHARS]

    # 2) Load chat history
    history = load_history(session_id, limit=3)

    history_text = ""
    for msg in history:
        prefix = "User" if msg["role"] == "user" else "Veronica"
        history_text += f"{prefix}: {msg['text']}\n"

    # 3) Final prompt
    final_prompt = (
        "You are Noah, the assistant for Christ Junior College.\n\n"
        "Context:\n"
        f"{context}\n\n"
        f"User question: {user_question}"
    )

    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config={
                "temperature": 0.7,

            }
        )

        # Rebuild Gemini chat with stored history
        gemini_history = [
            {
                "role": "user" if m["role"] == "user" else "model",
                "parts": [m["text"]],
            }
            for m in history
        ]

        chat = model.start_chat(history=gemini_history)
        response = chat.send_message(final_prompt)

        answer = (
            response.text.strip()
            if response and getattr(response, "text", None)
            else "Empty response from Gemini."
        )

        return answer

    except Exception as e:
        return f"Error calling Gemini: {e}"



# -------------------------------------------------
# Function to get Veronica's response based on KB
# -------------------------------------------------
def get_veronica_response(user_question: str, knowledge_base: Dict, session_id: str) -> str:
    # quick utility commands
    if user_question.lower() == 'date':
        answer = f"Today's date is {datetime.now().strftime('%Y-%m-%d')}"
        # store in history
        save_message(session_id, "user", user_question)
        save_message(session_id, "assistant", answer)
        return answer

    if user_question.lower() == 'time':
        answer = f"The current time is {datetime.now().strftime('%H:%M:%S')}"
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
        answer = get_gemini_response(user_question, session_id)

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
