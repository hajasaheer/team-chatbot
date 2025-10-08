import streamlit as st
from utils.auth import login, logout

st.set_page_config(page_title="Team Chatbot", layout="wide")

if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
    login()
else:
    st.sidebar.title("Menu")
    st.sidebar.success(f"Logged in as: {st.session_state['username']}")
    if st.sidebar.button("Logout"):
        logout()

    st.title("🤖 Team Chatbot")
    st.write("Welcome to the AI Team Chatbot! Use the sidebar to navigate.")
    st.write("👉 Go to Chatbot or Admin page using the sidebar menu.")
