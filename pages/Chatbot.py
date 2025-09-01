import streamlit as st
from utils.qa import (
    exact_cache_get, semantic_cache_get, exact_cache_set,
    qdrant_search, build_prompt, stream_ollama_answer
)

st.title("💬 Chatbot")

query = st.text_input("Ask a question about your documents")

if query:
    # Check cache
    cached = exact_cache_get(query)
    if cached:
        st.info("From cache")
        st.write(cached)
    else:
        # Retrieve context
        results = qdrant_search(query)
        docs = results["documents"][0]
        context = "\n\n".join(docs)

        # Build prompt
        prompt = build_prompt(context, query)

        # Stream response
        st.write("🤖 Answer:")
        response_text = ""
        resp_container = st.empty()
        for chunk in stream_ollama_answer(prompt, "llama3"):
            response_text += chunk
            resp_container.markdown(response_text)

        # Cache result
        exact_cache_set(query, response_text)
