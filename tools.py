from duckduckgo_search import DDGS

def search_tool(query):
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=5)
        return [r["body"] for r in results]