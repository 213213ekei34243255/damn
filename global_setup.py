# global_setup.py (run once at app start)
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

model_embed = SentenceTransformer("all-MiniLM-L6-v2")

def load_memory_and_precompute(memory_path="veronica_memory.json", emb_path="chunk_embs.npy"):
    mem_file = Path(memory_path)

    if not mem_file.exists():
        return [], np.array([])

    data = json.loads(mem_file.read_text(encoding="utf-8"))

    # 🔥 FIX: support BOTH formats (chunks OR structured JSON)
    if "chunks" in data:
        chunks = data.get("chunks", [])
    else:
        # convert structured JSON → text chunks
        chunks = [json.dumps(data)]

    if not chunks:
        return [], np.array([])

    chunk_embs = model_embed.encode(chunks, convert_to_tensor=False)

    np.save(emb_path, chunk_embs)

    return chunks, chunk_embs

# At app startup:
CHUNKS, CHUNK_EMBS = load_memory_and_precompute()
