
from rags import BioRAG

from prompts import biodiversity_qa_tpl 

rag = BioRAG(store_path="biodiversity_store", data_dir="biodiversity_data", qa_prompt_tpl=biodiversity_qa_tpl)
# rag = BioRAG

def is_covered_by_biodiversity(query: str, min_doc_chars: int = 300, min_docs: int = 2, top_k: int = 4) -> bool:
    """
    Heuristic: require at least `min_docs` retrieved docs and at least one doc with >= min_doc_chars.
    This reduces false positives from tiny snippets.
    """
    try:
        docs = rag.retrieve_docs(query, top_k=top_k)
        if not docs or len(docs) < min_docs:
            return False

        best_len = 0
        total_len = 0
        for d in docs:
            if isinstance(d, str):
                text = d
            else:
                # try common attrs
                text = getattr(d, "get_text", lambda: None)()
                if text is None:
                    text = getattr(d, "text", None) or getattr(d, "source_text", None) or ""
            l = len((text or "").strip())
            total_len += l
            best_len = max(best_len, l)

        # Strict: need one long doc and at least min_docs present
        if best_len >= min_doc_chars and len(docs) >= min_docs:
            return True
        return False
    except Exception:
        return False
    
def is_covered_by_biodiversity_with_scores(query: str, min_score: float = 0.7, min_docs: int = 2, top_k: int = 4) -> bool:
    """
    Use node.score or node.metadata['score'] if present.
    Accept if at least min_docs have score >= min_score, or the best score >= min_score.
    Score scale depends on retriever: often cosine-similarity-like in [0,1].
    """
    try:
        docs = rag.retrieve_docs(query, top_k=top_k)
        if not docs:
            return False

        scores = []
        for d in docs:
            # try common score attributes
            s = None
            s = getattr(d, "score", None)
            if s is None:
                # some nodes keep metadata dict
                meta = getattr(d, "metadata", {}) or {}
                s = meta.get("score") or meta.get("similarity_score") or meta.get("similarity")
            # if still None, try node.extra_info or get_score method
            if s is None:
                s = getattr(d, "get_score", lambda: None)()
            if s is None:
                # last resort: length-based fallback
                text = getattr(d, "get_text", lambda: None)() or getattr(d, "text", "") or ""
                s = min(0.0, len(text) / 2000)  # tiny fallback; will be <1
            scores.append(float(s))

        # sort descending: best scores first
        scores.sort(reverse=True)
        # require best score >= min_score AND at least min_docs above a slightly lower threshold
        if scores and scores[0] >= min_score and sum(1 for sc in scores if sc >= (min_score * 0.9)) >= min_docs:
            return True
        return False
    except Exception:
        return False

import numpy as np

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def is_covered_by_biodiversity_by_embedding(query: str, top_k: int = 4, min_sim: float = 0.72, min_docs_above: int = 1) -> bool:
    """
    Use direct embedding cosine similarity between query and top_k docs.
    min_sim: 0.7-0.8 recommended for strict coverage.
    """
    try:
        # assume `embed_model` exists in rags module or Settings.embed_model
        from rags import embed_model  # or import from module where you set embed_model
        # embed query
        q_emb = embed_model.get_text_embedding(query)
        # retrieve docs (nodes)
        docs = rag.retrieve_docs(query, top_k=top_k)
        if not docs:
            return False

        sims = []
        for d in docs:
            text = ""
            if isinstance(d, str):
                text = d
            else:
                text = getattr(d, "get_text", lambda: None)() or getattr(d, "text", "") or ""
            if not text:
                continue
            # get embedding for doc text (you might want to chunk long docs)
            doc_emb = embed_model.get_text_embedding(text)
            s = cosine_sim(np.array(q_emb), np.array(doc_emb))
            sims.append(s)

        sims.sort(reverse=True)
        if len(sims) >= min_docs_above and sum(1 for s in sims if s >= min_sim) >= min_docs_above:
            return True
        return False
    except Exception as e:
        print("[Embedding coverage check failed]", e)
        return False

def decide_biodiversity_coverage(query):

    q = query.lower()

    if is_covered_by_biodiversity_by_embedding(query, min_sim=0.8, min_docs_above=1):
        print("By Embeddings")
        return True
    if is_covered_by_biodiversity_with_scores(query, min_score=0.8, min_docs=1):
        print("By Scores")
        return True
  
    return False

# print('============')
# print("Where is Mexico?")
# docs = rag.retrieve_docs("Where is Mexico?", top_k=5)
# for i,d in enumerate(docs):
#     print("=== DOC", i, "type:", type(d))
#     print("text preview:", getattr(d, "get_text", lambda: d[:300])()[:400])
#     print("score:", getattr(d, "score", None), "metadata:", getattr(d, "metadata", None))
#     print("-----")