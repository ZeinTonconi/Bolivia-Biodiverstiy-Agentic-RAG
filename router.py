
from rags import BioRAG

from prompts import biodiversity_qa_tpl 

rag = BioRAG(store_path="biodiversity_store", data_dir="biodiversity_data", qa_prompt_tpl=biodiversity_qa_tpl)

import numpy as np

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def is_covered_by_biodiversity_by_embedding(query: str, top_k: int = 4, min_sim: float = 0.8, min_docs_above: int = 1) -> bool:
    try:
        from rags import embed_model  
        q_emb = embed_model.get_text_embedding(query)
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

OTHER_COUNTRIES = [
            "peru", "perú", "chile", "argentina", "brasil", "brazil", "rusia", "russia",
            "china", "usa", "united states", "mexico", "méxico", "colombia", "ecuador",
            "paraguay", "uruguay", "antartida", "antártida"
        ]

def decide_biodiversity_coverage(query):

    q = query.lower()

    if is_covered_by_biodiversity_by_embedding(query, min_sim=0.8, min_docs_above=2):
        print("By Embeddings")
        if not(any(country in q for country in OTHER_COUNTRIES)):
            return True
  
    return False
