import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def search_web(query):
    try:
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=3
        )
        
        results = []
        for r in response["results"]:
            results.append(r["content"])
        
        return "\n\n".join(results)
    
    except Exception as e:
        print(f"Tavily error: {e}")
        return ""

if __name__ == "__main__":
    result = search_web("grounding techniques for dissociation crisis counseling")
    print(result)