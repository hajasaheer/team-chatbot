import streamlit as st
from utils.ingestion import ingest_uploaded_file
from utils.qa import ensure_collections, _collection_count, clear_cache, DOC_COLLECTION, CACHE_COLLECTION

from utils.qa import qclient, DOC_COLLECTION
import streamlit as st

with st.expander("🔍 Qdrant Debug"):
    try:
        scroll_res, _ = qclient.scroll(collection_name=DOC_COLLECTION, limit=5)
        for pt in scroll_res:
            st.json(pt.payload)
    except Exception as e:
        st.error(f"Error scrolling Qdrant: {e}")


st.title("📂 Admin – Document Management")

# Ensure Qdrant collections exist
ensure_collections()

# File uploader
uploaded_file = st.file_uploader("Upload a document", type=["txt", "md", "pdf", "docx", "csv"])
if uploaded_file is not None:
    with st.spinner("Indexing document..."):
        success, fname = ingest_uploaded_file(uploaded_file)
    if success:
        st.success(f"✅ Indexed: {fname}")
    else:
        st.info(f"ℹ️ Already ingested: {fname}")

# Collection counts
st.subheader("📊 Collection Stats")
st.write(f"**{DOC_COLLECTION}**: {_collection_count(DOC_COLLECTION)} vectors")
st.write(f"**{CACHE_COLLECTION}**: {_collection_count(CACHE_COLLECTION)} cache entries")

# Cache clearing
if st.button("Clear Cache"):
    clear_cache()
    st.success("Cache cleared!")
