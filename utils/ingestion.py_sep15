import os
import io
import hashlib
import streamlit as st
from typing import List, Tuple, Optional
from PyPDF2 import PdfReader
from qdrant_client.http import models as qmodels
import uuid

from utils.qa import qclient, EMBED_MODEL, DOC_COLLECTION, file_sha256, embed_texts_parallel


CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF file bytes."""
    text = ""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        st.error(f"Failed to parse PDF: {e}")
    return text

def make_point_id(digest: str, idx: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{digest}_{idx}"))

def ingest_uploaded_file(uploaded_file) -> Tuple[bool, Optional[str]]:
    """
    Ingest a new uploaded file into Qdrant.
    Returns (success, filename).
    """
    file_bytes = uploaded_file.read()
    digest = file_sha256(file_bytes)
    fname = uploaded_file.name

    # 🔎 Check if this exact file is already in Qdrant
    try:
        existing = qclient.scroll(
            collection_name=DOC_COLLECTION,
            scroll_filter=qmodels.Filter(
                must=[qmodels.FieldCondition(key="file_digest", match=qmodels.MatchValue(value=digest))]
            ),
            limit=1
        )
        if existing and existing[0]:
            st.warning(f"File {fname} already ingested (sha256 match).")
            return False, fname
    except Exception:
        pass

    # Parse document
    text = extract_text_from_pdf(file_bytes)
    if not text.strip():
        st.error(f"No text could be extracted from {fname}")
        return False, fname

    # Chunk & embed
    chunks = chunk_text(text)
    embeddings = embed_texts_parallel(chunks)

    # Upsert chunks into Qdrant
    points = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        points.append(
            qmodels.PointStruct(
                id=make_point_id(digest, i),   # ✅ valid UUID
                vector=emb,
                payload={
                    "text": chunk,
                    "source_file": fname,
                    "chunk_index": i,
                    "file_digest": digest,
                },
            )
        )


    try:
        qclient.upsert(collection_name=DOC_COLLECTION, points=points)
        st.success(f"Ingested {len(points)} chunks from {fname}")
        return True, fname
    except Exception as e:
        st.error(f"Failed to upsert {fname}: {e}")
        return False, fname
