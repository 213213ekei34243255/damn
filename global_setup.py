import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# -----------------------------
# MODEL INIT
# -----------------------------
model_embed = SentenceTransformer("BAAI/bge-base-en-v1.5")


# -----------------------------
# LOAD JSON
# -----------------------------
def load_json(memory_path="veronica_memory.json"):
    mem_file = Path(memory_path)

    if not mem_file.exists():
        raise FileNotFoundError(f"{memory_path} not found")

    return json.loads(mem_file.read_text(encoding="utf-8"))


# -----------------------------
# FLATTEN JSON
# -----------------------------
def flatten_json(data):
    chunks = []

    def recurse(obj, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_path = f"{path}.{k}" if path else k
                recurse(v, new_path)

        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                recurse(item, f"{path}[{i}]")

        else:
            chunks.append({
                "text": f"{path}: {obj}",
                "path": path
            })

    recurse(data)
    return chunks


# -----------------------------
# DEDUPLICATE
# -----------------------------
def deduplicate_chunks(chunks):
    seen = set()
    unique = []

    for c in chunks:
        if c["text"] not in seen:
            seen.add(c["text"])
            unique.append(c)

    return unique

# -----------------------------
# 📘 DEFINITIONS LOGIC
# -----------------------------
def get_definition(query, data):
    defs = data.get("definitions", {})
    q = query.lower()

    # detect if user wants detailed
    is_detailed = any(word in q for word in ["detail", "full", "explain", "all", "subjects"])

    # check keys
    for key in defs:
        if key in q:
            # if detailed version exists
            if is_detailed and f"{key}_full" in defs:
                return defs[f"{key}_full"]
            return defs[key]

    return None
# -----------------------------
# EMBEDDINGS
# -----------------------------
def compute_embeddings(chunks, emb_path="chunk_embs.npy", force_recompute=False):
    emb_file = Path(emb_path)

    if emb_file.exists() and not force_recompute:
        return np.load(emb_path)

    texts = [c["text"] for c in chunks]

    embs = model_embed.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    np.save(emb_path, embs)
    return embs


# -----------------------------
# SEARCH
# -----------------------------
def search_memory(query, chunks, chunk_embs, top_k=15):
    query_emb = model_embed.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    scores = np.dot(chunk_embs, query_emb)
    top_indices = np.argsort(scores)[-top_k:][::-1]

    return [
        {
            "text": chunks[i]["text"],
            "score": float(scores[i])
        }
        for i in top_indices
    ]


# -----------------------------
# 🔥 MAPPING
# -----------------------------
def resolve_stream_mapping(query, mappings):
    query = query.lower()

    for key, streams in mappings.items():
        if key in query:
            return streams

    return None


# -----------------------------
# 💰 FEES LOGIC
# -----------------------------
def get_stream_fees(query, data):
    mappings = data.get("mappings", {})
    fees = data.get("fees", {})

    streams = resolve_stream_mapping(query, mappings)

    if not streams:
        return None

    results = []
    for s in streams:
        if s in fees:
            results.append(f"{s} costs ₹{fees[s]} per year")

    return results if results else None


# -----------------------------
# 🧠 RESPONSE FORMATTER (RULES APPLIED)
# -----------------------------
def format_response(text, data):
    if not text:
        return ""

    # Use "we" tone
    text = text.replace("The college", "We").replace("the college", "we")

    # Split into sentences
    sentences = text.split(".")
    sentences = [s.strip() for s in sentences if s.strip()]

    # Keep 2–4 sentences
    

    formatted = ". ".join(sentences)

    if not formatted.endswith("."):
        formatted += "."

    return formatted


# -----------------------------
# 🚀 MAIN RESPONSE FUNCTION
# -----------------------------
def get_response(query, data, chunks, chunk_embs):
    q = query.lower()

    # ---- DEFINITIONS FIRST ----
    definition = get_definition(query, data)
    if definition:
        return format_response(definition, data)

    # ---- FEES PRIORITY ----
    if "fee" in q or "fees" in q:
        fee_result = get_stream_fees(query, data)
        if fee_result:
            text = ". ".join(fee_result)
            return format_response(text, data)

    # ---- ADMISSION ----
    if "admission" in q or "1st puc" in q or "puc open" in q:
        pu_status = data["admissions"]["pu"]["status"]
        faq_note = data["faq"]["admission_status"]
        return format_response(f"{pu_status}. {faq_note}", data)

    # ---- HOSTEL ----
    if "hostel" in q:
        return format_response(data["faq"]["hostel"], data)

    # ---- FALLBACK SEARCH ----
    results = search_memory(query, chunks, chunk_embs)
    text = " ".join(results)

    return format_response(text, data)


# -----------------------------
# STARTUP
# -----------------------------
DATA = load_json()

CHUNKS = deduplicate_chunks(flatten_json(DATA))
CHUNK_EMBS = compute_embeddings(CHUNKS)
