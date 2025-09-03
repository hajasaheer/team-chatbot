import streamlit as st
from utils.qa import (
    exact_cache_get, semantic_cache_get, exact_cache_set,
    semantic_cache_set, qdrant_search, build_prompt, stream_ollama_answer
)

st.title("💬 Chatbot")

query = st.text_input("Ask a question about your documents")

if query:
    # 1️⃣ Try exact cache first
    cached = exact_cache_get(query)

    # 2️⃣ If not found, try semantic cache
    if not cached:
        sem_cache = semantic_cache_get(query)
        if sem_cache:
            cached, score = sem_cache
            st.info(f"From semantic cache (score={score:.2f})")

    # 3️⃣ If cache hit → display
    if cached:
        st.write(cached)

    else:
        # 🔍 Retrieve context from Qdrant
        results = qdrant_search(query)
        if not results:
            st.warning("No relevant documents found.")
        else:
            docs = [r.payload.get("text", "") for r in results if r.payload]
            context = "\n\n".join(docs)

            # 📝 Build prompt (fixed arg order: query, then context)
            prompt = build_prompt(query, context)

            # 🤖 Stream response
            st.write("🤖 Answer:")
            response_text = ""
            resp_container = st.empty()
            for chunk in stream_ollama_answer(prompt, "llama3"):
                response_text += chunk
                resp_container.markdown(response_text)

            # 💾 Cache result in both exact + semantic cache
            exact_cache_set(query, response_text)
            semantic_cache_set(query, response_text)
