from agents import planner_agent, research_agent, analyst_agent, writer_agent
from tools import search_tool

def run_super_chain(user_input):
    plan = planner_agent(user_input)
    research = research_agent(user_input, search_tool)
    analysis = analyst_agent(research)
    final_output = writer_agent(analysis)

    return {
        "plan": plan,
        "research": research,
        "analysis": analysis,
        "final": final_output
    }