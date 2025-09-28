# tools.py
from crewai.tools import tool
import requests
from crewai_tools import LlamaIndexTool
from config import get_agent_settings
from rags import BioRAG, bioRAG

SETTINGS = get_agent_settings()

bio_description = """
El Motor de Consultas de Biodiversidad es una herramienta impulsada por IA que proporciona información fáctica extraída de documentos de biodiversidad seleccionados, estudios de campo e informes ecológicos de la región objetivo.
Capacidades incluyen:
- Listas de especies e información taxonómica reportadas en los estudios e inventarios indexados.
- Resúmenes de tipos de hábitats, zonas ecológicas y notas de distribución.
- Amenazas documentadas, estado de conservación y tendencias históricas de población encontradas en los informes fuente.
- Datos clave de localidades (sitios, coordenadas, áreas protegidas) cuando estén presentes en los documentos.
Uso:
Haz preguntas sobre ocurrencia de especies, hallazgos de estudios, hábitats, problemas de conservación u otros hechos presentes en los documentos de biodiversidad seleccionados. La herramienta responderá utilizando únicamente las fuentes indexadas.
Nota:
Usa esta herramienta para la recuperación fáctica de los documentos de biodiversidad seleccionados.
"""

biodiversity_rag_tool = LlamaIndexTool.from_query_engine(
    bioRAG.get_query_engine(),
    name="Bolivia biodiversity guide",
    description=bio_description
)

# engine = bioRAG.get_query_engine() 

# def bio_query_raw(input_data):
#     """Search the information in the book"""
#     # print("[DEBUG] bio_query_raw normalized query:", input_data[:300])  
#     res = engine.query(input_data)
#     # print("[DEBUG] engine returned (truncated):", res[:500])
#     return res

# bio_query_tool = tool("Biodiversity Book")(bio_query_raw)

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


