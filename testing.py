# quick_test.py
from dotenv import load_dotenv
load_dotenv()

from rags import BioRAG
from prompts import biodiversity_qa_tpl  # spanish template

rag = BioRAG(store_path="biodiversity_store", data_dir="biodiversity_data", qa_prompt_tpl=biodiversity_qa_tpl)
qe = rag.get_query_engine()
q = "¿Qué anfibios hay Bolivia?"
print(qe.query(q))
