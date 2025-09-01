import os
import hashlib
from datetime import datetime
from typing import Optional, Tuple, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer

# Ollama
try:
    import ollama
    OLLAMA_NATIVE = True
except Exception:
    OLLAMA_NATIVE = False

try:
    from langchain_ollama import ChatOllama, OllamaLLM
except Exception:
    ChatOllama = None
    OllamaLLM = None


# -----------------------------
# Config
# -----------------------------
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
DOC_COLLECTION = os.getenv("DOC_COLLECTION", "documents")
CACHE_COLLECTION = os.getenv("CACHE_COLLECTION", "chat_cache")

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama3")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
SEMANTIC_CACHE_MIN_SCORE = float(os.getenv("SEMANTIC_CACHE_MIN_SCORE", "0.50"))

# -----------------------------
# Embedding model + Qdrant
# -----------------------------
st.info("Loading embedding model…")
EMBED_MODEL = SentenceTransformer(EMBED_MODEL_NAME)
EMBED_DIM = EMBED_MODEL.get_sentence_embedding_dimension()
st.success(f"Embedding model loaded ({EMBED_DIM} dims)")

qclient = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# -----------------------------
# Helpers
# -----------------------------
def file_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# -----------------------------
# Qdrant setup
# -----------------------------
def _collection_needs_recreate(name: str, want_size: int) -> bool:
    try:
        info = qclient.get_collection(collection_name=name)
        stored = int(info.config.params.vectors.size)  # type: ignore
        return stored != int(want_size)
    except Exception:
        return True

def ensure_collections():
    if _collection_needs_recreate(DOC_COLLECTION, EMBED_DIM):
        qclient.recreate_collection(
            collection_name=DOC_COLLECTION,
            vectors_config=qmodels.VectorParams(size=EMBED_DIM, distance=qmodels.Distance.COSINE)
        )
    if _collection_needs_recreate(CACHE_COLLECTION, EMBED_DIM):
        qclient.recreate_collection(
            collection_name=CACHE_COLLECTION,
            vectors_config=qmodels.VectorParams(size=EMBED_DIM, distance=qmodels.Distance.COSINE)
        )

def _collection_count(name: str) -> int:
    try:
        return int(qclient.count(collection_name=name, exact=True).count)
    except Exception:
        return 0


# -----------------------------
# Embeddings (parallel)
# -----------------------------
MAX_WORKERS = max(1, min(16, (os.cpu_count() or 1)))

def embed_texts_parallel(texts: List[str], max_workers: int = MAX_WORKERS) -> List[List[float]]:
    embeddings: List[Optional[List[float]]] = [None] * len(texts)
    def worker(i, txt):
        v = EMBED_MODEL.encode(txt, show_progress_bar=False, convert_to_numpy=True)
        return i, v.tolist()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(worker, i, t): i for i, t in enumerate(texts)}
        for fut in as_completed(futures):
            try:
                i, vec = fut.result()
                embeddings[i] = vec
            except Exception:
                embeddings[futures[fut]] = EMBED_MODEL.encode(texts[futures[fut]]).tolist()
    for idx, e in enumerate(embeddings):
        if e is None:
            embeddings[idx] = EMBED_MODEL.encode(texts[idx]).tolist()
    return embeddings


# -----------------------------
# Qdrant search
# -----------------------------

def qdrant_search(query: str, top_k: int = 6) -> Dict[str, List]:
    qvec = EMBED_MODEL.encode(query, convert_to_numpy=True).tolist()
    try:
        results = qclient.search(
            collection_name=DOC_COLLECTION,
            query_vector=qvec,
            limit=top_k,
            with_payload=True,
            with_vectors=False
        )
    except Exception as e:
        st.error(f"Qdrant search failed: {e}")
        return {"documents": [[]], "metadatas": [[]], "scores": [[]]}

    documents, metadatas, scores = [], [], []
    for r in results:
        payload = r.payload or {}
        text = payload.get("text", "")
        metadata = {k: v for k, v in payload.items() if k != "text"}

        # Build proper reference with real filename + chunk id
        ref = f"[{metadata.get('source_file', 'unknown')}#chunk{metadata.get('chunk_index', '?')}]"

        # Append the reference inline so LLM always sees it
        documents.append(f"{text}\n(Source: {ref})")
        metadatas.append(metadata)
        scores.append(float(getattr(r, "score", 0.0)))

    return {"documents": [documents], "metadatas": [metadatas], "scores": [scores]}


# -----------------------------
# Cache helpers
# -----------------------------
def exact_cache_get(query: str) -> Optional[str]:
    key = sha256_text(query)
    try:
        point = qclient.retrieve(collection_name=CACHE_COLLECTION, ids=[key])
        if point and len(point) > 0:
            p = point[0]
            if p.payload and "response" in p.payload:
                return p.payload["response"]
    except Exception:
        pass
    return None

def exact_cache_set(query: str, response: str):
    key = sha256_text(query)
    vec = EMBED_MODEL.encode(query, convert_to_numpy=True).tolist()
    payload = {"query": query, "response": response, "ts": datetime.utcnow().isoformat()}
    pt = qmodels.PointStruct(id=key, vector=vec, payload=payload)
    try:
        qclient.upsert(collection_name=CACHE_COLLECTION, points=[pt])
    except Exception:
        pass

def semantic_cache_get(query: str, top_k: int = 1, min_score: float = SEMANTIC_CACHE_MIN_SCORE) -> Optional[Tuple[str, float]]:
    qvec = EMBED_MODEL.encode(query, convert_to_numpy=True).tolist()
    try:
        res = qclient.search(collection_name=CACHE_COLLECTION, query_vector=qvec, limit=top_k, with_payload=True)
        if res:
            item = res[0]
            payload = item.payload or {}
            response_text = payload.get("response")
            score = float(getattr(item, "score", 0.0))
            if response_text and score >= min_score:
                return response_text, score
    except Exception:
        pass
    return None

def semantic_cache_set(query: str, response: str):
    exact_cache_set(query, response)

def clear_cache() -> bool:
    try:
        qclient.delete_collection(collection_name=CACHE_COLLECTION)
    except Exception:
        pass
    qclient.recreate_collection(
        collection_name=CACHE_COLLECTION,
        vectors_config=qmodels.VectorParams(size=EMBED_DIM, distance=qmodels.Distance.COSINE)
    )
    return True


# -----------------------------
# Ollama streaming
# -----------------------------
def stream_ollama_answer(prompt: str, model_name: str, options=None):
    options = options or {}
    if OLLAMA_NATIVE:
        try:
            for part in ollama.generate(model=model_name, prompt=prompt, stream=True, options=options):
                chunk = part.get("response", "")
                if chunk:
                    yield chunk
            return
        except Exception:
            pass
    if OllamaLLM is not None:
        try:
            llm = OllamaLLM(model=model_name, options=options)
            text = llm.invoke(prompt)
            for i in range(0, len(text), 64):
                yield text[i:i+64]
            return
        except Exception:
            pass
    if ChatOllama is not None:
        try:
            llm = ChatOllama(model=model_name, **(options or {}))
            for chunk in llm.stream(prompt):
                yield getattr(chunk, "content", str(chunk))
            return
        except Exception:
            pass
    yield "Ollama backend unavailable."


# -----------------------------
# Prompt builder
# -----------------------------
def build_prompt(context: str, question: str) -> str:
    return (
        "You are a precise assistant that ONLY answers using the provided context.\n"
        "Rules:\n"
        " - If the answer is not in the context, reply exactly: \"I don’t know based on the uploaded documents.\"\n"
        " - Never guess or invent information.\n"
        " - Always cite the document names and chunk IDs that appear in the context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

