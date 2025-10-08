import streamlit as st
import bcrypt
import os
from dotenv import load_dotenv

load_dotenv()  # Load from .env

USERNAME = os.getenv("APP_USERNAME")
PASSWORD_HASH = os.getenv("APP_PASSWORD_HASH")

def verify_user(username: str, password: str) -> bool:
    """Verify username and password."""
    if username != USERNAME:
        return False
    if PASSWORD_HASH is None:
        return False
    return bcrypt.checkpw(password.encode(), PASSWORD_HASH.encode())

def login():
    """Display login form."""
    st.title("🔐 Login to Team Chatbot")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if verify_user(username, password):
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid username or password")

def logout():
    """Clear session."""
    for key in ["logged_in", "username"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

def login_required(func):
    """Decorator to restrict access to logged-in users."""
    def wrapper(*args, **kwargs):
        if not st.session_state.get("logged_in", False):
            st.warning("Please log in to continue.")
            login()
            return
        return func(*args, **kwargs)
    return wrapper
