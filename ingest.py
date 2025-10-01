
import os
from rags import BioRAG 
DATA_DIR = "biodiversity_data"  
STORE_DIR = "biodiversity_store" 

def main():
    print("DATA_DIR =", DATA_DIR)
    print("STORE_DIR =", STORE_DIR)

    if os.path.exists(STORE_DIR):
        print(f"[INFO] Store directory '{STORE_DIR}' already exists. Loading existing index.")
        rag = BioRAG(store_path=STORE_DIR, data_dir=None, qa_prompt_tpl=None)
        print("[INFO] Index loaded from existing store.")
        return

    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    print("[INFO] Creating new BioRAG index. This may take a while depending on file sizes.")
    rag = BioRAG(store_path=STORE_DIR, data_dir=DATA_DIR, qa_prompt_tpl=None)
    print("[INFO] Ingest finished. Index persisted to:", STORE_DIR)

if __name__ == "__main__":
    main()
