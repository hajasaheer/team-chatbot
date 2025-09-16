import streamlit as st

st.set_page_config(page_title="Team Knowledge Chatbot", layout="wide")

st.title("📚 Team Knowledge Chatbot")
st.write("Welcome! Use the sidebar to navigate:")

st.markdown(
    """
    - 👉 **Chatbot** — Ask questions to the AI
    - 🔑 **Admin** — Manage documents & settings
    """
)
