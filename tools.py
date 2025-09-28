# tools.py
from crewai.tools import tool
import requests
from crewai_tools import LlamaIndexTool
from config import get_agent_settings
from rags import BioRAG, bioRAG
from typing import Optional, Any
import logging
import threading

SETTINGS = get_agent_settings()

# bio_description = """
# El Motor de Consultas de Biodiversidad es una herramienta impulsada por IA que proporciona información fáctica extraída de documentos de biodiversidad seleccionados, estudios de campo e informes ecológicos de la región objetivo.
# Capacidades incluyen:
# - Listas de especies e información taxonómica reportadas en los estudios e inventarios indexados.
# - Resúmenes de tipos de hábitats, zonas ecológicas y notas de distribución.
# - Amenazas documentadas, estado de conservación y tendencias históricas de población encontradas en los informes fuente.
# - Datos clave de localidades (sitios, coordenadas, áreas protegidas) cuando estén presentes en los documentos.
# Uso:
# Haz preguntas sobre ocurrencia de especies, hallazgos de estudios, hábitats, problemas de conservación u otros hechos presentes en los documentos de biodiversidad seleccionados. La herramienta responderá utilizando únicamente las fuentes indexadas.
# Nota:
# Usa esta herramienta para la recuperación fáctica de los documentos de biodiversidad seleccionados.
# """

# biodiversity_rag_tool = LlamaIndexTool.from_query_engine(
#     bioRAG.get_query_engine(),
#     name="Bolivia biodiversity guide",
#     description=bio_description
# )

logger = logging.getLogger(__name__)
_query_lock = threading.Lock()

def _extract_query_from_payload(payload: Any) -> str:
    """
    Accepts: str, dict, or object. Returns a query string or empty string if none found.
    """
    if payload is None:
        return ""
    # plain string
    if isinstance(payload, str):
        return payload.strip()
    # dictionary-like payload
    if isinstance(payload, dict):
        # common keys we might receive
        for k in ("query", "text", "description", "prompt"):
            v = payload.get(k)
            if v:
                return str(v).strip()
        # fallback: maybe the tool was given a Document-like dict with 'content' or 'source'
        for k in ("content", "source_text", "source"):
            v = payload.get(k)
            if v:
                return str(v).strip()
        return ""
    # fallback: try to access common attributes
    for attr in ("query", "description", "text", "content"):
        v = getattr(payload, attr, None)
        if v:
            return str(v).strip()
    # last resort: str(payload)
    return str(payload).strip()

@tool("Bolivia biodiversity guide")
def biodiversity_rag_tool_wrapper(
    query: Optional[str] = None,
    **kwargs
) -> str:
    """
    Tool wrapper that normalizes incoming payloads to a query string and then runs the BioRAG
    query engine. This avoids Pydantic validation errors when the runtime passes `description`
    or other shapes instead of `query`.
    """
    try:
        # extract a query from either explicit param or kwargs (or fallback to kwargs payload)
        if query and isinstance(query, str) and query.strip():
            q = query.strip()
        else:
            # maybe caller passed 'description' or the entire task as a dict in kwargs
            # try common locations:
            q = ""
            # direct fallback: if the caller packed the whole payload into a key like 'payload'
            if "payload" in kwargs:
                q = _extract_query_from_payload(kwargs["payload"])
            if not q:
                # check other kwargs commonly used by crew/agent runtimes
                q = _extract_query_from_payload(kwargs.get("description") or kwargs.get("text") or kwargs.get("input") or kwargs)

        if not q:
            raise ValueError("No query text found in tool invocation. Provide 'query' or 'description'.")

        # thread-safe access to the query engine (helps avoid subtle concurrency problems)
        with _query_lock:
            qe = bioRAG.get_query_engine()
            # call the query engine - adapt to your LlamaIndex version; using qe.query is general
            resp = qe.query(q)

        # Convert response to text - many LL Index responses implement __str__
        # If you prefer a structured dict you can parse resp.source_nodes etc.
        return str(resp)

    except Exception as e:
        # keep failures informative for debugging (crew will show this in the agent/tool error)
        logger.exception("biodiversity_rag_tool_wrapper failed")
        return f"[ToolError] {type(e).__name__}: {e}"

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


