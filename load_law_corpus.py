# load_law_corpus.py
import os
import csv
import pandas as pd
import chromadb
from dotenv import load_dotenv
import google.generativeai as genai
import time

# --- CONFIGURATION ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

genai.configure(api_key=GOOGLE_API_KEY)
CHROMA_PATH = "fir_vector_db"
LAW_COLLECTION_NAME = "laws_collection"
EMBEDDING_MODEL = "models/text-embedding-004"
IPC_CSV_FILE_PATH = "ipc.csv"
CRPC_CSV_FILE_PATH = "crpc.csv"

# --- HELPER FUNCTION FOR EMBEDDING ---
def embed_text_gai(text, model):
    """Generates embedding for a given text using Google AI."""
    try:
        result = genai.embed_content(model=model, content=text, task_type="RETRIEVAL_DOCUMENT")
        return result['embedding']
    except Exception as e:
        print(f"Error embedding text: {text[:50]}... Error: {e}")
        time.sleep(1) # Rate limit handling
        return embed_text_gai(text, model) # Retry

# --- MAIN SCRIPT ---
def main():
    print("Initializing ChromaDB client...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # 1. ===== LOAD LAWS COLLECTION =====
    print(f"\n--- Loading IPC & CrPC Sections into '{LAW_COLLECTION_NAME}' ---")
    
    # Delete collection if it exists to ensure a fresh start
    if LAW_COLLECTION_NAME in [c.name for c in client.list_collections()]:
        print(f"Collection '{LAW_COLLECTION_NAME}' already exists. Deleting it.")
        client.delete_collection(name=LAW_COLLECTION_NAME)

    law_collection = client.create_collection(name=LAW_COLLECTION_NAME)

    # Load IPC
    try:
        # Files are TSV-like with quoted, multi-line descriptions; be permissive
        df_ipc = pd.read_csv(
            IPC_CSV_FILE_PATH,
            sep='\t',
            engine='python',
            quotechar='"',
            doublequote=True,
            escapechar='\\',
            on_bad_lines='warn',
            encoding='utf-8',
            keep_default_na=False,
        )
    except FileNotFoundError:
        print(f"Error: The file '{IPC_CSV_FILE_PATH}' was not found.")
        return

    # Load CrPC
    try:
        df_crpc = pd.read_csv(
            CRPC_CSV_FILE_PATH,
            sep='\t',
            engine='python',
            quotechar='"',
            doublequote=True,
            escapechar='\\',
            on_bad_lines='warn',
            encoding='utf-8',
            keep_default_na=False,
        )
    except FileNotFoundError:
        print(f"Error: The file '{CRPC_CSV_FILE_PATH}' was not found.")
        return

    documents: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    # Build IPC documents
    print(f"Found {len(df_ipc)} IPC sections to process from '{IPC_CSV_FILE_PATH}'.")
    for idx, row in df_ipc.iterrows():
        section_raw = str(row.get('Section', '')).strip()
        section_number = section_raw.replace('_', ' ').replace('IPC', 'IPC').strip() or 'IPC'
        section_title = str(row.get('Offense', '')).strip()
        description = str(row.get('Description', '')).strip()
        punishment = str(row.get('Punishment', '')).strip()
        full_text = description + (f"\nPunishment: {punishment}" if punishment else "")

        doc_content = f"Section {section_number}: {section_title}. Text: {full_text}"
        documents.append(doc_content)
        metadatas.append({
            'code': 'IPC',
            'section_number': section_number,
            'section_name': section_title,
            'full_text': full_text,
            'url': ''
        })
        ids.append(f"ipc_{idx}")

    # Build CrPC documents
    print(f"Found {len(df_crpc)} CrPC sections to process from '{CRPC_CSV_FILE_PATH}'.")
    for idx, row in df_crpc.iterrows():
        sec_num = str(row.get('Section', '')).strip()
        section_number = f"CrPC {sec_num}" if sec_num else "CrPC"
        # Column name in file appears as 'Section _name' with a space
        section_title = str(row.get('Section _name', '')).strip()
        description = str(row.get('Description', '')).strip()
        full_text = description

        doc_content = f"Section {section_number}: {section_title}. Text: {full_text}"
        documents.append(doc_content)
        metadatas.append({
            'code': 'CrPC',
            'section_number': section_number,
            'section_name': section_title,
            'full_text': full_text,
            'url': ''
        })
        ids.append(f"crpc_{idx}")

    # Embed and add to Chroma in batches for stability
    print("Embedding and adding IPC & CrPC sections to the collection in batches...")
    batch_size = 100
    total = len(documents)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        print(f"- Processing {start+1}-{end} / {total}")
        batch_docs = documents[start:end]
        batch_metas = metadatas[start:end]
        batch_ids = ids[start:end]
        batch_embeddings = [embed_text_gai(doc, EMBEDDING_MODEL) for doc in batch_docs]
        law_collection.add(
            embeddings=batch_embeddings,
            documents=batch_docs,
            metadatas=batch_metas,
            ids=batch_ids,
        )
    print(f"Successfully added {law_collection.count()} documents to '{LAW_COLLECTION_NAME}'.")
    print("\n✅ Corpus loading complete (IPC + CrPC only).")

if __name__ == "__main__":
    main()