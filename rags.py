import os
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    SimpleDirectoryReader,
    PromptTemplate,
    Settings,
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from config import get_agent_settings
from prompts import biodiversity_qa_tpl
import torch 

SETTINGS = get_agent_settings()

llm = OpenAI(model=SETTINGS.openai_model)
device = "cuda" if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 7 else "cpu"
print(f"[INFO] Embeddings running on: {device}")

embed_model = HuggingFaceEmbedding(
    model_name=SETTINGS.hf_embeddings_model,
    device=device
)

Settings.embed_model = embed_model
Settings.llm = llm


class BioRAG:
    def __init__(
        self,
        store_path: str,
        data_dir: str | None = None,
        qa_prompt_tpl: PromptTemplate | None = None,
    ):
        self.store_path = store_path

        if not os.path.exists(store_path) and data_dir is not None:
            self.index = self.ingest_data(store_path, data_dir)
        else:
            self.index = load_index_from_storage(
                StorageContext.from_defaults(persist_dir=store_path)
            )

        self.qa_prompt_tpl = qa_prompt_tpl

    def ingest_data(self, store_path: str, data_dir: str) -> VectorStoreIndex:
        print("Ingesting data...")
        print("Store Path: ", store_path)
        print("Data Dir: ", data_dir)
        documents = SimpleDirectoryReader(data_dir).load_data()
        index = VectorStoreIndex.from_documents(documents, show_progress=True)
        index.storage_context.persist(persist_dir=store_path)
        return index

    def get_query_engine(self) -> RetrieverQueryEngine:
        query_engine = self.index.as_query_engine()

        if self.qa_prompt_tpl is not None:
            query_engine.update_prompts(
                {"response_synthesizer:text_qa_template": self.qa_prompt_tpl}
            )

        return query_engine

    def retrieve_docs(self, query: str, top_k: int = 3):
        try:
            retriever = self.index.as_retriever(search_kwargs={"k": top_k})
            docs = retriever.retrieve(query)
            return docs
        except Exception:
            try:
                qe = self.index.as_query_engine()
                retr = getattr(qe, "retriever", None)
                if retr is not None and hasattr(retr, "retrieve"):
                    return retr.retrieve(query)[:top_k]
            except Exception:
                pass

        try:
            qe = self.index.as_query_engine()
            resp = qe.query(query)
            nodes = getattr(resp, "source_nodes", None) or getattr(resp, "nodes", None) or []
            return nodes[:top_k]
        except Exception:
            return []

bioRAG =  BioRAG(
        store_path="biodiversity_store", 
        data_dir="biodiversity_data",
        qa_prompt_tpl=biodiversity_qa_tpl
    )