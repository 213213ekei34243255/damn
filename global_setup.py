# global_setup.py (run once at app start)
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

model_embed = SentenceTransformer("all-MiniLM-L6-v2")

def load_memory_and_precompute(memory_path="veronica_memory.json", emb_path="chunk_embs.npy"):
    mem_file = Path(memory_path)
    data = json.loads(mem_file.read_text(encoding="utf-8"))
    chunks = data.get("chunks", [])
    # compute & save embeddings only if not present
    emb_file = Path(emb_path)
    if emb_file.exists():
        chunk_embs = np.load(emb_path)
    else:
        chunk_embs = model_embed.encode(chunks, convert_to_tensor=False)  # returns numpy array
        np.save(emb_path, chunk_embs)
    return chunks, chunk_embs

# At app startup:
CHUNKS, CHUNK_EMBS = load_memory_and_precompute()
