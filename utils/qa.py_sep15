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
DOC_COLLECTION = os.getenv("DOC_COLLECTION", "documents")
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

def make_cache_id(query: str) -> str:
    """Consistent cache ID for queries"""
    return sha256_text(query)

def make_uuid_from_hash(text: str) -> str:
    # Create a deterministic UUID from the SHA256 hash
    return str(uuid.UUID(hashlib.sha256(text.encode()).hexdigest()[0:32]))
# =========================
# Ollama Streaming
# =========================
def stream_ollama_answer(
    prompt: str,
    model: str = DEFAULT_MODEL,
    stream: bool = True,
):
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
        return int(info.config.params.vectors.size)  # type: ignore
    except Exception:
        return None

def _create_collection(name: str):
    qclient.create_collection(
        collection_name=name,
        vectors_config=qmodels.VectorParams(
            size=EMBED_DIM, distance=qmodels.Distance.COSINE
        ),
    )

def ensure_collections():
    for name in [DOC_COLLECTION, CACHE_COLLECTION]:
        try:
            qclient.get_collection(name)
            print(f"[Collections] {name} exists ✅")
        except Exception:
            print(f"[Collections] Creating {name} ✅")
            qclient.create_collection(
                collection_name=name,
                vectors_config=qmodels.VectorParams(
                    size=EMBED_DIM, distance=qmodels.Distance.COSINE
                ),
            )


    # Cache collection
    size = _get_collection_vector_size(CACHE_COLLECTION)
    if size is None:
        print(f"[Qdrant] Creating collection: {CACHE_COLLECTION}")
        _create_collection(CACHE_COLLECTION)
    elif size != EMBED_DIM:
        print(
            f"[Qdrant][WARN] {CACHE_COLLECTION} has vector size {size}, expected {EMBED_DIM}. "
            "Not recreating to avoid data loss."
        )

# =========================
# Search
# =========================
def qdrant_search(query: str, top_k: int = 5, score_threshold: float = 0.2):
    """Search document collection in Qdrant."""
    vec = EMBED_MODEL.encode(query, convert_to_numpy=True).tolist()
    try:
        res = qclient.search(
            collection_name=DOC_COLLECTION,
            query_vector=vec,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )
        return res
    except Exception as e:
        print(f"[Qdrant] Search failed: {e}")
        return []

        print(f"[Cache] Semantic cache set failed: {e}")


# =========================
# Cache Handling
# =========================

def exact_cache_get(query: str) -> Optional[str]:
    key = make_uuid_from_hash(query)   # ✅ use UUID consistently
    try:
        point = qclient.retrieve(collection_name=CACHE_COLLECTION, ids=[str(key)])
        if point and point[0].payload:
            resp = point[0].payload.get("response")
            if resp:
                print(f"[Cache] Exact hit id={str(key)[:8]}")
                return resp
    except Exception as e:
        print(f"[Cache] Exact retrieval failed: {e}")
    return None


def _cache_upsert_common(key: str, query: str, response: str, vector: List[float]):
    payload = {
        "query": query,
        "response": response,
        "ts": datetime.utcnow().isoformat(),
    }
    pt = qmodels.PointStruct(id=str(key), vector=vector, payload=payload)
    qclient.upsert(collection_name=CACHE_COLLECTION, points=[pt])
    print(f"[Cache] Upsert id={str(key)[:8]} query='{query[:40]}...'")



def exact_cache_set(query: str, response: str):
    key = make_uuid_from_hash(query)   # ✅ UUID instead of raw sha256 hex
    vec = EMBED_MODEL.encode(query, convert_to_numpy=True).tolist()
    payload = {
        "query": query,
        "response": response,
        "ts": datetime.utcnow().isoformat(),
    }
    print(f"[Cache] Upserting (exact) into {CACHE_COLLECTION} → id={key} query='{query[:30]}...'")
    pt = qmodels.PointStruct(id=key, vector=vec, payload=payload)
    try:
        qclient.upsert(collection_name=CACHE_COLLECTION, points=[pt])
    except Exception as e:
        print(f"[Cache] Upsert failed: {e}")

def semantic_cache_get(
    query: str, top_k: int = 1, min_score: float = SEMANTIC_CACHE_MIN_SCORE
) -> Optional[Tuple[str, float]]:
    qvec = EMBED_MODEL.encode(query, convert_to_numpy=True).tolist()
    try:
        res = qclient.search(
            collection_name=CACHE_COLLECTION,
            query_vector=qvec,
            limit=top_k,
            with_payload=True,
        )
        if res:
            item = res[0]
            response_text = item.payload.get("response") if item.payload else None
            score = float(getattr(item, "score", 0.0))
            if response_text and score >= min_score:
                print(f"[Cache] Semantic hit score={score:.3f}")
                return response_text, score
    except Exception as e:
        print(f"[Cache] Semantic lookup failed: {e}")
    return None




def semantic_cache_set(query: str, answer: str):
    vec = embed_texts_parallel([query])[0]
    uid = make_uuid_from_hash(query + answer)   # ✅ UUID instead of raw sha256 hex
    print(f"[Cache] Upserting (semantic) into {CACHE_COLLECTION} → id={uid[:8]} query='{query[:30]}...'")
    pt = qmodels.PointStruct(
        id=uid,
        vector=vec,
        payload={
            "query": query,
            "response": answer,
            "ts": datetime.utcnow().isoformat(),
        },
    )
    try:
        qclient.upsert(collection_name=CACHE_COLLECTION, points=[pt])
    except Exception as e:
        print(f"[Cache] Semantic cache set failed: {e}")

def delete_cache_entry(query: str) -> bool:
    key = make_uuid_from_hash(query)
    try:
        qclient.delete(
            collection_name=CACHE_COLLECTION,
            points_selector=qmodels.PointIdsList(points=[key]),
        )
        print(f"[Cache] Deleted id={key[:8]}")
        return True
    except Exception as e:
        print(f"[Cache] Delete failed: {e}")
        return False

# =========================
# Admin Helpers
# =========================
def _collection_count(name: str) -> int:
    try:
        res = qclient.count(collection_name=name, exact=True)
        return int(res.count)
    except Exception as e:
        print(f"[Qdrant] Count check failed for {name}: {e}")
        return 0

def clear_cache() -> bool:
    """
    Clear ALL cache entries.
    Safe to recreate here because it's an explicit admin action.
    """
    try:
        qclient.recreate_collection(
            collection_name=CACHE_COLLECTION,
            vectors_config=qmodels.VectorParams(
                size=EMBED_DIM, distance=qmodels.Distance.COSINE
            ),
        )
        print("[Cache] Cleared (recreated collection).")
        return True
    except Exception as e:
        print(f"[Cache] Clear cache failed: {e}")
        return False

# =========================
# Embedding helper
# =========================
def embed_texts_parallel(texts: List[str]) -> List[List[float]]:
    return EMBED_MODEL.encode(texts, convert_to_numpy=True).tolist()

# =========================
# Prompt Builder
# =========================
def build_prompt(query: str, context: str) -> str:
    """Builds the prompt (query first, then context)."""
    return f"""You are a helpful assistant. Use the provided context to answer the question.

Question:
{query}

Context:
{context}

Answer:"""

# Ensure collections exist
ensure_collections()
