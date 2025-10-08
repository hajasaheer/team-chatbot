import streamlit as st
from utils.qa import (
    exact_cache_get, semantic_cache_get, exact_cache_set,
    semantic_cache_set, qdrant_search, build_prompt, stream_ollama_answer
)


client_id = st.selectbox("Select Client", ["Common", "client1", "client2", "client3", "client4"])

st.title("💬 Chatbot")

query = st.text_input("Ask a question about your documents")

if query:
    cached = exact_cache_get(query)  # (optional: make cache per-client too)

    if not cached:
        sem_cache = semantic_cache_get(query)
        if sem_cache:
            cached, score = sem_cache
            st.info(f"From semantic cache (score={score:.2f})")

    if cached:
        st.write(cached)
    else:
        results = qdrant_search(query, client_id=client_id)   # ✅ pass client_id
        if not results:
            st.warning("No relevant documents found.")
        else:
            docs = [r.payload.get("text", "") for r in results if r.payload]
            context = "\n\n".join(docs)

            prompt = build_prompt(query, context)

            st.write("🤖 Answer:")
            response_text = ""
            resp_container = st.empty()
            for chunk in stream_ollama_answer(prompt, "llama3"):
                response_text += chunk
                resp_container.markdown(response_text)

            exact_cache_set(query, response_text)
            semantic_cache_set(query, response_text)

