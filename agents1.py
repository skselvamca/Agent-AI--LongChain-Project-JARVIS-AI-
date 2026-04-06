import os
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant"     # ⚡ super fast
)

def ask_llm(prompt):
    return llm.invoke(prompt).content


# Planner Agent
def planner_agent(task):
    return ask_llm(f"Break this task into step-by-step plan:\n{task}")


# Research Agent
def research_agent(task, tool):
    results = tool(task)
    return "\n".join(results)


# Analyst Agent
def analyst_agent(data):
    return ask_llm(f"Analyze this data and extract key insights:\n{data}")


# Writer Agent
def writer_agent(insights):
    return ask_llm(f"Write a clean, professional final answer:\n{insights}")

print("API KEY:", os.getenv("GROQ_API_KEY"))