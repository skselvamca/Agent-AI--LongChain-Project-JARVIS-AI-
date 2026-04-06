import streamlit as st
from autonomous_agent import autonomous_agent
from file_reader import read_pdf

st.set_page_config(page_title="JARVIS AI", layout="wide")

st.title("🤖 JARVIS AI (Groq Powered)")

# Input
user_input = st.text_area("Enter your task:", height=150)

# File upload
uploaded_file = st.file_uploader("📂 Upload PDF (optional)")

if uploaded_file:
    file_text = read_pdf(uploaded_file)
    user_input += "\n" + file_text
    st.success("PDF content added to task")

# Run button
if st.button("🚀 Run JARVIS"):
    if not user_input.strip():
        st.warning("Please enter a task")
    else:
        with st.spinner("JARVIS is thinking... 🤖"):
            result = autonomous_agent(user_input)

        st.subheader("🧠 Thinking Steps")
        for i, step in enumerate(result["steps"], 1):
            st.write(f"Step {i}:")
            st.write(step)

        st.subheader("✅ Final Answer")
        st.success(result["final"])