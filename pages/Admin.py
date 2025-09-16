# pages/Admin.py
import streamlit as st
from utils.ingestion import ingest_uploaded_file
from utils.qa import (
    ensure_client_collection,
    ensure_cache_collection,
    clear_cache,
    delete_cache_entry,
    qclient,
    DOC_COLLECTION,
    CACHE_COLLECTION,
)

st.title("📂 Admin – Document Management")

# Ensure Qdrant collections exist
ensure_cache_collection()

# Select client
client_id = st.selectbox("Select Client", ["Common", "Bosswallah", "SBSOL", "Thinksynq", "Pumex"])
collection_name = f"documents_{client_id}"
ensure_client_collection(client_id)

# ----------------------------
# File uploader
# ----------------------------
uploaded_file = st.file_uploader("Upload a document", type=["txt", "md", "pdf", "docx", "csv"])
if uploaded_file:
    with st.spinner(f"Indexing into {client_id}..."):
        success, fname = ingest_uploaded_file(uploaded_file, client_id=client_id)
    if success:
        st.success(f"✅ Indexed: {fname} for {client_id}")
    else:
        st.info(f"ℹ️ Already ingested: {fname}")

# ----------------------------
# Collection stats
# ----------------------------
st.subheader("📊 Collection Stats")
try:
    doc_count = qclient.count(collection_name=collection_name).count
except Exception:
    doc_count = 0

try:
    cache_count = qclient.count(collection_name=CACHE_COLLECTION).count
except Exception:
    cache_count = 0

st.write(f"**{collection_name}**: {doc_count} vectors")
st.write(f"**{CACHE_COLLECTION}**: {cache_count} cache entries")

# ----------------------------
# Cache actions
# ----------------------------
st.subheader("🧹 Cache Management")

if st.button("Clear Cache"):
    clear_cache()
    st.success("Cache cleared!")

st.write("Delete specific cache entry:")
query_to_delete = st.text_input("Enter the exact question to delete from cache:")
if st.button("Delete Cache Entry"):
    if query_to_delete.strip():
        ok = delete_cache_entry(query_to_delete.strip())
        if ok:
            st.success(f"Cache entry for '{query_to_delete}' deleted.")
        else:
            st.error("Failed to delete cache entry.")

# ----------------------------
# Clear all uploaded documents for a client
# ----------------------------
st.subheader("🗑️ Clear Uploaded Documents")

if st.button(f"Clear all documents for {client_id}"):
    try:
        # Recreate the collection, deleting all uploaded documents
        qclient.recreate_collection(
            collection_name=collection_name,
            vectors_config={
                "size": 384,  # adjust based on embedding model
                "distance": "Cosine",
            },
        )
        st.success(f"All uploaded documents for {client_id} cleared!")
    except Exception as e:
        st.error(f"Failed to clear documents for {client_id}: {e}")

