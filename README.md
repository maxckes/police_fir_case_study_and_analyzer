## Police FIR Analysis Assistant

An interactive Streamlit application that helps Indian police officers analyze FIR (First Information Report) text. The app retrieves potentially relevant IPC/CrPC sections from a local vector database (ChromaDB) using Google AI embeddings and then asks a Gemini model to produce a structured, bilingual (English/Telugu) analysis with recommended sections, rationales, quotes, and actionable steps.

### Features
- **FIR input**: Paste text or upload a `.pdf`/`.txt` file. PDFs are parsed with PyPDF2.
- **Vector retrieval (RAG)**: Retrieves top‑K relevant IPC/CrPC sections from a local ChromaDB collection populated from `ipc.csv` and `crpc.csv`.
- **LLM analysis (Gemini)**: Uses `gemini-2.5-flash` to generate a strict JSON object with:
  - Recommended sections with confidence
  - English and Telugu rationales and verbatim law quotes
  - English and Telugu actionable steps
  - FIR excerpts that justify each recommendation
- **Bilingual UX**: English and Telugu display, with graceful Telugu fallbacks to English when translation is unavailable.
- **Downloadable output**: Option to display and download the raw JSON.

### Project structure
```
policefir/
  app.py                 # Streamlit UI and end-to-end workflow
  load_law_corpus.py     # Loads IPC/CrPC data into Chroma with embeddings
  requirements.txt       # Python dependencies
  ipc.csv                # IPC sections (tab-separated)
  crpc.csv               # CrPC sections (tab-separated)
  fir_vector_db/         # Chroma persistent store (created by loader)
```

### Prerequisites
- Python 3.10+ recommended
- A Google AI Studio API key with access to Gemini models
- Internet access (for embeddings and LLM calls)

### Installation
1. Create and activate a virtual environment.
   - macOS/Linux:
     ```bash
     python3 -m venv .venv && source .venv/bin/activate
     ```
   - Windows (PowerShell):
     ```powershell
     python -m venv .venv; .venv\Scripts\Activate.ps1
     ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root with your API key:
   ```env
   GOOGLE_API_KEY=your_api_key_here
   ```

### Load the law corpus (build the vector DB)
The app expects a Chroma collection named `laws_collection` at `fir_vector_db/`.

- The loader reads two TSV-like files with permissive parsing:
  - `ipc.csv` expects columns: `Section`, `Offense`, `Description`, `Punishment`.
  - `crpc.csv` expects columns: `Section`, `Section _name`, `Description`.
- Embeddings are created with `models/text-embedding-004` and stored as a persistent Chroma collection.

Run the loader:
```bash
python load_law_corpus.py
```
Notes:
- If the collection `laws_collection` already exists, the loader deletes and recreates it.
- The loader runs in batches and retries on transient embedding errors.

### Run the app
```bash
streamlit run app.py
```
Then open the local URL shown in your terminal. In the sidebar you can:
- Select jurisdiction (display hint used in the prompt)
- Choose Top‑K law sections to retrieve
- Toggle “Show Raw LLM Output” to inspect/download the JSON
- Upload a `.pdf`/`.txt` FIR or paste text in the chat input

### How it works (high‑level)
1. Text comes from user input or PDF/TXT upload.
2. A query embedding is generated with `models/text-embedding-004`.
3. Top‑K IPC/CrPC sections are retrieved from Chroma (metadata only).
4. A detailed system prompt is assembled containing:
   - FIR summary
   - Retrieved law sections (section number, name, full text)
   - Jurisdiction
   - A strict JSON schema the model must follow
5. `gemini-2.5-flash` returns a JSON object that the UI renders and can export.

### Configuration knobs
- In `app.py`:
  - `LLM_MODEL_NAME = "gemini-2.5-flash"`
  - `EMBEDDING_MODEL_NAME = "models/text-embedding-004"`
  - `CHROMA_PATH = "fir_vector_db"`
  - `LAW_COLLECTION_NAME = "laws_collection"`
  - Generation controls: `LLM_TEMPERATURE`, `MAX_OUTPUT_TOKENS`
- In `load_law_corpus.py`:
  - `CHROMA_PATH`, `LAW_COLLECTION_NAME`, `EMBEDDING_MODEL`

To switch models or paths, change the constants above and re-run the loader if embeddings change.

### Updating or extending the corpus
- Edit or replace `ipc.csv` / `crpc.csv` with your sources (keep the expected columns).
- Re-run `python load_law_corpus.py` to rebuild embeddings and the collection.
- You may add URLs in metadata later and teach the UI to display them (currently empty).

### Troubleshooting
- “Google API Key not found” in the app: Ensure `.env` has `GOOGLE_API_KEY` and the process can read it.
- “Failed to connect to the Vector Database”: Run `python load_law_corpus.py` first to create the collection.
- “Error reading PDF”: The file might be scanned or have complex layout; try uploading a `.txt` version.
- Embedding/LLM rate limits: The loader retries on errors with a short delay. If it loops too long, re-run later.
- Empty or poor results: Increase Top‑K in the sidebar, verify your CSV contents, and rebuild the corpus.

### Privacy and usage
- The UI shows a disclaimer and is intended for internal police use. Do not input sensitive PII.
- The app sends text to Google services for embeddings and generation.

### Development notes
- Streamlit caching is used for:
  - `@st.cache_resource` for the persistent Chroma connection
  - `@st.cache_data` for retrievals
- UI includes confidence badges and bilingual fallbacks (Telugu falls back to English if needed).

### Commands quick reference
```bash
# Build vector DB
python load_law_corpus.py

# Run app
streamlit run app.py

# (Optional) Upgrade pip tooling
python -m pip install --upgrade pip setuptools wheel
```

### License
No license file detected. Consider adding one (e.g., MIT, Apache-2.0) if you plan to share or open-source.


 
