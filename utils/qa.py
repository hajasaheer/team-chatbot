# utils/qa.py

import os
import requests
import hashlib
import json
from datetime import datetime
from typing import Optional, List, Tuple
import uuid

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from dotenv import load_dotenv

load_dotenv()

# =========================
# Config
# =========================
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
DOC_COLLECTION = os.getenv("DOC_COLLECTION", "documents_common")
CACHE_COLLECTION = os.getenv("CACHE_COLLECTION", "chat_cache")

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama3")
SEMANTIC_CACHE_MIN_SCORE = float(os.getenv("SEMANTIC_CACHE_MIN_SCORE", "0.75"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Init model + client
EMBED_MODEL = SentenceTransformer(EMBED_MODEL_NAME)
EMBED_DIM = EMBED_MODEL.get_sentence_embedding_dimension()
qclient = QdrantClient(url=QDRANT_URL)

# =========================
# Helpers
# =========================
def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def file_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def make_uuid_from_hash(text: str) -> str:
    """Deterministic UUID from hash (for cache IDs)."""
    return str(uuid.UUID(hashlib.sha256(text.encode()).hexdigest()[0:32]))

# =========================
# Ollama Streaming
# =========================
def stream_ollama_answer(prompt: str, model: str = DEFAULT_MODEL, stream: bool = True):
    """Stream answer tokens from Ollama API."""
    url = f"{OLLAMA_URL}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": stream}

    try:
        with requests.post(url, json=payload, stream=True, timeout=300) as r:
            for line in r.iter_lines():
                if line:
                    try:
                        obj = json.loads(line.decode("utf-8"))
                        if "response" in obj:
                            yield obj["response"]
                    except Exception:
                        continue
    except Exception as e:
        yield f"[Error contacting Ollama: {e}]"

# =========================
# Collection Management
# =========================
def _get_collection_vector_size(name: str) -> Optional[int]:
    try:
        info = qclient.get_collection(collection_name=name)
        return int(info.config.params.vectors.size)
    except Exception:
        return None

def _create_collection(name: str):
    qclient.create_collection(
        collection_name=name,
        vectors_config=qmodels.VectorParams(size=EMBED_DIM, distance=qmodels.Distance.COSINE),
    )

def ensure_client_collection(client_id: str) -> str:
    collection_name = f"documents_{client_id}"
    try:
        qclient.get_collection(collection_name)
    except Exception:
        print(f"[Qdrant] Creating collection: {collection_name}")
        _create_collection(collection_name)
    return collection_name

def ensure_cache_collection():
    size = _get_collection_vector_size(CACHE_COLLECTION)
    if size is None:
        print(f"[Qdrant] Creating cache collection: {CACHE_COLLECTION}")
        _create_collection(CACHE_COLLECTION)
    elif size != EMBED_DIM:
        print(f"[Qdrant][WARN] Cache vector size mismatch ({size} vs {EMBED_DIM})")

# =========================
# Search
# =========================
def qdrant_search(query: str, client_id: str, top_k: int = 5, score_threshold: float = 0.2):
    """Search inside a specific client's collection."""
    vec = EMBED_MODEL.encode(query, convert_to_numpy=True).tolist()
    collection_name = ensure_client_collection(client_id)
    try:
        res = qclient.search(
            collection_name=collection_name,
            query_vector=vec,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )
        return res
    except Exception as e:
        print(f"[Qdrant] Search failed for {collection_name}: {e}")
        return []

# =========================
# Cache Handling
# =========================
def exact_cache_get(query: str, client_id: str) -> Optional[str]:
    key = make_uuid_from_hash(client_id + ":" + query)
    try:
        point = qclient.retrieve(collection_name=CACHE_COLLECTION, ids=[str(key)])
        if point and point[0].payload and point[0].payload.get("client_id") == client_id:
            resp = point[0].payload.get("response")
            if resp:
                print(f"[Cache] Exact hit for client={client_id} id={str(key)[:8]}")
                return resp
    except Exception as e:
        print(f"[Cache] Exact retrieval failed: {e}")
    return None

def exact_cache_set(query: str, response: str, client_id: str):
    key = make_uuid_from_hash(client_id + ":" + query)
    vec = EMBED_MODEL.encode(query, convert_to_numpy=True).tolist()
    pt = qmodels.PointStruct(
        id=key,
        vector=vec,
        payload={
            "query": query,
            "response": response,
            "client_id": client_id,
            "ts": datetime.utcnow().isoformat(),
        },
    )
    try:
        qclient.upsert(collection_name=CACHE_COLLECTION, points=[pt])
        print(f"[Cache] Upserted exact id={key[:8]} client={client_id}")
    except Exception as e:
        print(f"[Cache] Upsert failed: {e}")

def semantic_cache_get(query: str, client_id: str, top_k: int = 1,
                       min_score: float = SEMANTIC_CACHE_MIN_SCORE) -> Optional[Tuple[str, float]]:
    qvec = EMBED_MODEL.encode(query, convert_to_numpy=True).tolist()
    try:
        res = qclient.search(
            collection_name=CACHE_COLLECTION,
            query_vector=qvec,
            limit=top_k,
            with_payload=True,
            query_filter=qmodels.Filter(
                must=[qmodels.FieldCondition(key="client_id", match=qmodels.MatchValue(value=client_id))]
            ),
        )
        if res:
            item = res[0]
            response_text = item.payload.get("response") if item.payload else None
            score = float(getattr(item, "score", 0.0))
            if response_text and score >= min_score:
                print(f"[Cache] Semantic hit client={client_id} score={score:.3f}")
                return response_text, score
    except Exception as e:
        print(f"[Cache] Semantic lookup failed: {e}")
    return None

def semantic_cache_set(query: str, answer: str, client_id: str):
    vec = EMBED_MODEL.encode(query, convert_to_numpy=True).tolist()
    uid = make_uuid_from_hash(client_id + ":" + query + ":" + answer)
    pt = qmodels.PointStruct(
        id=uid,
        vector=vec,
        payload={
            "query": query,
            "response": answer,
            "client_id": client_id,
            "ts": datetime.utcnow().isoformat(),
        },
    )
    try:
        qclient.upsert(collection_name=CACHE_COLLECTION, points=[pt])
        print(f"[Cache] Semantic upsert client={client_id} id={uid[:8]}")
    except Exception as e:
        print(f"[Cache] Semantic cache set failed: {e}")


# =========================
# Embedding helper
# =========================
def embed_texts_parallel(texts: List[str]) -> List[List[float]]:
    """Encode a list of texts into embeddings (parallelized if supported by the model)."""
    return EMBED_MODEL.encode(texts, convert_to_numpy=True).tolist()

# =========================
# Prompt Builder
# =========================
def build_prompt(query: str, context: str) -> str:
    return f"""
You are an assistant that answers questions **strictly using only the provided context**.
If the context does not contain the answer, respond exactly with:
"I don't know based on the provided context."

Do not generate or assume any information that is not present in the context.
Do not start your answer with phrases like "According to the provided context" or similar.

---
Question:
{query}

---
Context:
{context or "No relevant context provided."}

---
Answer:
"""

# =========================
# Init on Import
# =========================
ensure_cache_collection()

