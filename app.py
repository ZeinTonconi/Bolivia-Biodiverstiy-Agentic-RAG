# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import traceback
from main import init_system, answer_query  # import the init & runner

app = FastAPI(title="Biodiversity RAG (Crew) API", version="0.1")

# initialize at import / startup
SYSTEM = init_system()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    tool: str
    answer: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    try:
        q = req.query.strip()
        if not q:
            raise HTTPException(status_code=400, detail="Empty query")
        out = answer_query(SYSTEM, q)
        return QueryResponse(tool=out["tool"], answer=out["answer"])
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
