import streamlit as st
from langchain_groq import ChatGroq
from tools import search_tool

# ✅ Get API key safely
KEY = st.secrets.get("KEY")

if not KEY:
    st.error("❌ API KEY not found. Please check Streamlit Secrets.")
    st.stop()

# ✅ Memory store
memory = []

# ✅ LLM setup
llm = ChatGroq(
    groq_api_key=KEY,
    model_name="llama-3.1-8b-instant"
)

def ask_llm(prompt):
    try:
        return llm.invoke(prompt).content
    except Exception as e:
        return f"❌ LLM Error: {str(e)}"


def autonomous_agent(user_task, max_steps=5):
    global memory

    context = "\n".join(memory)

    for step in range(max_steps):

        prompt = f"""
You are JARVIS AI.

Task: {user_task}

Memory:
{context}

Decide next step:
- THINK
- SEARCH
- FINAL

Format:
ACTION:
CONTENT:
"""

        response = ask_llm(prompt)

        memory.append(response)
        context += "\n" + response

        # 🔍 Search action
        if "SEARCH" in response:
            query = response.split("CONTENT:")[-1].strip()
            try:
                results = search_tool(query)
                context += "\n".join(results)
            except Exception as e:
                context += f"\nSearch Error: {str(e)}"

        # ✅ Final output
        if "FINAL" in response:
            return {
                "steps": memory,
                "final": response
            }

    return {
        "steps": memory,
        "final": "⚠️ Stopped after max steps"
    }