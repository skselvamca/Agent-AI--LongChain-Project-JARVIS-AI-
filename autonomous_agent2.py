import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from tools import search_tool

load_dotenv()

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant"     # ⚡ super fast
)

def ask_llm(prompt):
    return llm.invoke(prompt).content


def autonomous_agent2(user_task, max_steps=5):
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

        if "SEARCH" in response:
            query = response.split("CONTENT:")[-1].strip()
            results = search_tool(query)
            context += "\n".join(results)

        if "FINAL" in response:
            return {
                "steps": memory,
                "final": response
            }

    return {"steps": memory, "final": "Stopped"}