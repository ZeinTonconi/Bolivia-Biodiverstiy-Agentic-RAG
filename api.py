import logging
import threading
import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
load_dotenv()
from router import decide_biodiversity_coverage
from tools import web_search_tool, biodiversity_rag_tool_wrapper
from rags import bioRAG 
from crewai import Agent, Crew, Task
from prompts import web_prompt_str

logger = logging.getLogger("agentic_rag_api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Agentic RAG (Crew pipelines)")

app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# serialize potentially-racy calls to the RAG/index/crew
_execution_lock = threading.Lock()

def _run_biodiversity_crew(user_query: str, max_attempts: int = 2, retry_delay: float = 0.25) -> str:
    """
    Create the biodiversity Agent and Task, kickoff the Crew, and return the textual result.
    Retries once if a transient tool validation error occurs (like the Pydantic 'query' missing error).
    """
    biodiversity_agent = Agent(
        role="Experto en Biodiversidad",
        goal="Proporciona información útil sobre la biodiversidad de Bolivia",
        backstory=(
            "Eres un especialista en estudios de biodiversidad e informes ecológicos. "
            "Muestra los hallazgos con precisión utilizando únicamente los documentos de biodiversidad seleccionados proporcionados. "
            "Das prioridad a la claridad, a la citación fáctica y no inventas detalles faltantes."
        ),
        tools=[biodiversity_rag_tool_wrapper],
    )

    get_info_task = Task(
        description=f"Obtén información precisa y relevante sobre la consulta del usuario utilizando la herramienta de guía de libros proporcionada. Consulta del usuario: {user_query}",
        expected_output="Un texto con detalles, hechos e información sobre la biodiversidad.",
        agent=biodiversity_agent,
    )

    attempts = 0
    last_exc = None
    while attempts < max_attempts:
        attempts += 1
        try:
            with _execution_lock:
                crew = Crew(agents=[biodiversity_agent], tasks=[get_info_task], verbose=True)
                result = crew.kickoff()
            return str(result).strip()
        except Exception as e:
            last_exc = e

            msg = str(e)
            logger.warning("Biodiversity crew failed (attempt %d/%d): %s", attempts, max_attempts, msg)

            if "Field required" in msg or "QueryToolSchema" in msg or "query" in msg.lower():
                if attempts < max_attempts:
                    time.sleep(retry_delay)
                    continue
            break

    logger.exception("Biodiversity crew failed after %d attempts", attempts)
    raise last_exc or RuntimeError("Biodiversity crew failed")


def _run_web_path(user_query: str) -> str:
    """
    Execute the web search path: get snippets via web_search_tool, then run a WebSynthesizer agent
    that synthesizes a short answer from the two snippets (same behavior as your main.py).
    """
    try:
        snippet_1, url_1, snippet_2, url_2 = web_search_tool.run(user_query)
    except Exception as e:
        logger.exception("web_search_tool failed")
        raise

    web_agent = Agent(
        role="WebSynthesizer",
        goal="Synthesize a short answer using only the two provided web snippets",
        backstory="You are concise and must strictly use only the two provided snippets.",
        tools=[],
        prompt_template=web_prompt_str,
        llm_kwargs={"temperature": 0, "max_tokens": 180},
    )

    web_task = Task(
        description=(
            f"User query: {user_query}\n\nSnippet 1:\n{snippet_1}\n\nSnippet 2:\n{snippet_2}"
        ),
        expected_output="Short answer restricted to the snippets provided.",
        agent=web_agent,
    )

    try:
        with _execution_lock:
            crew_web = Crew(agents=[web_agent], tasks=[web_task], verbose=True)
            web_result = crew_web.kickoff()
        model_answer = str(web_result).strip()
    except Exception as e:
        logger.exception("Web crew failed")
        raise

    final = (
        "Note: This information is outside the biodiversity guide and was gathered quickly from web sources.\n\n"
        + model_answer
        + "\n\nLinks:\n"
        + (url_1 or "")
        + (("\n" + url_2) if url_2 else "")
    )
    return final


class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    answer: str
    tool_used: str
    sources: Optional[List[str]] = None

class QueryRequest(BaseModel):
    query: str

class RouteResponse(BaseModel):
    chosen_tool: str


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    q = (req.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty query")
    
    try:
        covered = decide_biodiversity_coverage(q)
    except Exception:
        logger.exception("decide_biodiversity_coverage failed; defaulting to False")
        covered = False

    if covered:
        try:
            answer_text = _run_biodiversity_crew(q)
            return AskResponse(answer=answer_text, tool_used="Bolivia biodiversity guide", sources=None)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Biodiversity tool failed: {type(e).__name__}: {e}")
    else:
        try:
            answer_text = _run_web_path(q)
            return AskResponse(answer=answer_text, tool_used="Web Search", sources=None)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Web path failed: {type(e).__name__}: {e}")


@app.get("/status")
def status():
    rag_loaded = False
    doc_count = None
    try:
        idx = getattr(bioRAG, "index", None)
        if idx is not None:
            rag_loaded = True
            try:
                ds = getattr(idx, "docstore", None) or getattr(idx, "_docstore", None)
                if ds is not None:
                    if hasattr(ds, "num_docs"):
                        doc_count = int(getattr(ds, "num_docs"))
                    elif hasattr(ds, "get_count"):
                        doc_count = int(ds.get_count())
                    else:
                        entries = getattr(ds, "_docs", None) or getattr(ds, "_dict", None) or getattr(ds, "docs", None)
                        if entries is not None:
                            try:
                                doc_count = len(entries)
                            except Exception:
                                doc_count = None
            except Exception:
                doc_count = None
    except Exception:
        rag_loaded = False

    return {"status": "ok", "rag_loaded": rag_loaded, "doc_count": doc_count}


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/route", response_model=RouteResponse)
async def route(req: QueryRequest):
    """Check if query is covered by the Book Guide or needs Web Search."""
    if decide_biodiversity_coverage(req.query):
        chosen_tool = "Book Guide"
    else:
        chosen_tool = "Web Search"
    return RouteResponse(chosen_tool=chosen_tool)