# tools.py
from crewai.tools import tool
import requests
import os
from openai import OpenAI
from crewai_tools import LlamaIndexTool
from config import get_agent_settings
from rags import BioRAG
from prompts import travel_guide_qa_tpl

SETTINGS = get_agent_settings()

travel_guide_description = """
The Travel Guide Query Engine is an AI-powered tool that provides up-to-date travel advice and insights from a curated travel guidebook. Currently, it is based on the book Lonely Planet's: Bolivia, offering rich details to help plan your trip efficiently.

Capabilities include:
- Detailed highlights and itineraries to personalize your travel experience.
- Insider tips on saving time, avoiding crowds, and navigating like a local.
- Essential information on operating hours, websites, transit, and prices.
- Honest reviews across all budgets, covering food, sightseeing, shopping, and hidden gems.
- Cultural insights for a deeper understanding of local history, art, cuisine, and politics.
- Coverage of key destinations, such as La Paz, Lake Titicaca, Salar de Uyuni, and the Amazon Basin.

Simply input your travel questions or ask for recommendations in plain text, and the tool will provide accurate, context-rich responses to guide your journey.
Note:
    DO use this tool for recommendations, and general information retrieval. For ACTIONS use the specific 
    tool, flight, bus, hotel or restaurant.
"""

travel_guide_rag_tool = LlamaIndexTool.from_query_engine(
    BioRAG(
        store_path="travel_guide_store", 
        qa_prompt_tpl=travel_guide_qa_tpl
    ).get_query_engine(),
    name="Bolivia Travel guide",
    description=travel_guide_description
)

def web_search_raw(query: str) -> str:
    """Search the web using SerpAPI and return summaries of the top results."""
    api_key = SETTINGS.serper_api_key
    params = {
        "api_key": api_key,
        "engine": "google",
        "q": query,
        "google_domain": "google.com",
        "gl": "us",
        "hl": "en"
    }

    try:
        response = requests.get("https://serpapi.com/search", params=params)
        results = response.json()
    except Exception as e:
        return f"[WebSearchError] {type(e).__name__}: {e}"

    organic = results.get("organic_results", [])[:2]
    if not organic:
        return "[WebSearch] No results found."
    
    snippet_1 = organic[0].get("snippet","")
    url_1     = organic[0].get("link","")
    snippet_2 = organic[1].get("snippet","") if len(organic) > 1 else ""
    url_2     = organic[1].get("link","") if len(organic) > 1 else ""
    return (snippet_1, url_1, snippet_2, url_2)


# Decorated Tool object for Crew (uses the raw function)
web_search_tool = tool("Web Search")(web_search_raw)


