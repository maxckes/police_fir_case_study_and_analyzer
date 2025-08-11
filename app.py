# app.py
import streamlit as st
import google.generativeai as genai
import chromadb
import os
import json
from dotenv import load_dotenv

# --- PAGE CONFIGURATION & INITIALIZATION ---
st.set_page_config(
    page_title="Police FIR Analysis Assistant",
    page_icon="⚖️",
    layout="wide"
)

# Basic production styling for better readability and structure
_APP_STYLE = """
<style>
:root {
  --bg: #ffffff;
  --card-bg: #ffffff;
  --text: #111827;
  --muted: #6b7280;
  --border: rgba(0,0,0,0.08);
  --shadow: 0 1px 2px rgba(0,0,0,0.05);
  --quote-bg: #0e1117;
  --quote-text: #e6e6e6;

  --badge-green-bg: #e7f7ef; --badge-green-text: #127c51;
  --badge-amber-bg: #fff3e0; --badge-amber-text: #8a5a00;
  --badge-red-bg: #fdecea; --badge-red-text: #8a1c1c;
  --divider: rgba(0,0,0,0.08);
}

@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0b0f14;
    --card-bg: #111827;
    --text: #e5e7eb;
    --muted: #9ca3af;
    --border: rgba(255,255,255,0.08);
    --shadow: 0 1px 2px rgba(0,0,0,0.3);
    --quote-bg: #0b0f14;
    --quote-text: #e5e7eb;

    --badge-green-bg: #0f2e24; --badge-green-text: #34d399;
    --badge-amber-bg: #2a1e0a; --badge-amber-text: #fbbf24;
    --badge-red-bg: #2a0f0f; --badge-red-text: #f87171;
    --divider: rgba(255,255,255,0.08);
  }
}

.main > div { padding-top: 0.75rem; color: var(--text); }
.card { border: 1px solid var(--border); border-radius: 12px; padding: 1rem 1.25rem; background: var(--card-bg); box-shadow: var(--shadow); }
.badge { display:inline-block; padding:0.25rem 0.6rem; border-radius:999px; font-size:0.8rem; font-weight:600; margin-left:0.5rem; }
.badge-green { background:var(--badge-green-bg); color:var(--badge-green-text); }
.badge-amber { background:var(--badge-amber-bg); color:var(--badge-amber-text); }
.badge-red { background:var(--badge-red-bg); color:var(--badge-red-text); }
.quote-block { background:var(--quote-bg); color:var(--quote-text); padding:0.75rem 0.9rem; border-radius:8px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; white-space: pre-wrap; }
.divider { height:1px; background:var(--divider); margin:0.75rem 0 1rem; }
.small-muted { color: var(--muted); font-size: 0.85rem; }
</style>
"""
st.markdown(_APP_STYLE, unsafe_allow_html=True)

# Load environment variables and configure API
load_dotenv()
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
except AttributeError as e:
    st.error("🚨 Google API Key not found. Please set it in your .env file.", icon="🚨")
    st.stop()

# --- MODEL AND DATABASE SETUP ---
LLM_MODEL_NAME = "gemini-2.5-flash"
EMBEDDING_MODEL_NAME = "models/text-embedding-004"
CHROMA_PATH = "fir_vector_db"
LAW_COLLECTION_NAME = "laws_collection"

# Generation controls
LLM_TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 65535
@st.cache_resource(show_spinner=False)
def _get_law_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(LAW_COLLECTION_NAME)

# Connect to ChromaDB
try:
    law_collection = _get_law_collection()
except Exception as e:
    st.error(f"🚨 Failed to connect to the Vector Database. Have you run `load_law_corpus.py`? Error: {e}", icon="🚨")
    st.stop()

# Initialize session state
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "fir_text" not in st.session_state:
    st.session_state.fir_text = ""
if "messages" not in st.session_state:
    st.session_state.messages = []
if "jurisdiction" not in st.session_state:
    st.session_state.jurisdiction = "Telangana"
if "top_k_laws" not in st.session_state:
    st.session_state.top_k_laws = 5


# --- FUNCTION DEFINITIONS ---

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Splits text into chunks.
    Note: This is a placeholder as requested. For this app, we embed entire law sections,
    not chunks of the FIR. This function would be used if we were to index large case files.
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(text)

def _embed_query_text(text: str):
    """Embeds query text using the same model used to index the collections."""
    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL_NAME,
            content=text,
            task_type="RETRIEVAL_QUERY",
        )
        return result["embedding"]
    except Exception as e:
        st.error(f"Failed to embed query text: {e}")
        return None

@st.cache_data(show_spinner=False)
def translate_to_te(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    try:
        model = genai.GenerativeModel(
            LLM_MODEL_NAME,
            generation_config={
                "response_mime_type": "text/plain",
                "temperature": 0.0,
                "max_output_tokens": 2048,
            }
        )
        resp = model.generate_content(
            f"Translate to Telugu. Preserve legal meaning, names, and formatting. "
            f"Only output the translation without any extra commentary:\n\n{text}"
        )
        return (getattr(resp, "text", "") or "").strip()
    except Exception:
        return text

@st.cache_data(show_spinner=False)
def retrieve_law_sections(query_text, k=5):
    """Retrieves top-k relevant law sections from ChromaDB."""
    embedding = _embed_query_text(query_text)
    if embedding is None:
        return []
    results = law_collection.query(
        query_embeddings=[embedding],
        n_results=k,
        include=["metadatas"],
    )
    return results['metadatas'][0] if results and results.get('metadatas') else []

## Past case retrieval removed: using only IPC/CrPC as RAG context

def build_system_prompt(fir_summary, relevant_laws, jurisdiction):
    """Constructs the detailed system prompt for the Gemini API call."""
    laws_context = "\n".join([
        f"- Section {law['section_number']} ({law['section_name']}): {law['full_text']}"
        for law in relevant_laws
    ]) if relevant_laws else "No relevant laws found."

    json_schema = {
        "sections": [
            {
                "section": "IPC/CrPC Section Number (e.g., IPC 420 or CrPC 41)",
                "rationale_en": "2-3 sentence rationale in English.",
                "rationale_te": "2-3 sentence rationale in Telugu.",
                "reasoning_quote_en": "Verbatim quote from the provided law text supporting the recommendation.",
                "reasoning_quote_te": "చట్ట పాఠ్యంలోని సంబంధిత వాక్యాన్ని తెలుగు లో యథాతధంగా ఇవ్వండి; తెలుగు లేనప్పుడు ఆంగ్ల వాక్యానికి నమ్మకమైన తెలుగు అనువాదాన్ని ఇవ్వండి.",
                "fir_quote_en": "Verbatim excerpt from the FIR summary that supports this recommendation.",
                "fir_quote_te": "కేసు వివరాల్లోని సంబంధిత వాక్యాన్ని తెలుగు లో యథాతధంగా ఇవ్వండి; అది తెలుగు లో లేకపోతే నమ్మకమైన తెలుగు అనువాదాన్ని ఇవ్వండి.",
                "law_citation": {
                    "title_en": "Official name of the section in English",
                    "title_te": "తెలుగు శీర్షిక; అందుబాటులో లేకపోతే ఆంగ్ల శీర్షికను నమ్మకంగా తెలుగు లోకి అనువదించండి.",
                    "full_text_en": "Complete official text in English (or best available English summary).",
                    "full_text_te": "తెలుగు పూర్తి పాఠ్యం; అందుబాటులో లేకపోతే ఆంగ్ల పాఠ్యానికి నమ్మకమైన తెలుగు అనువాదాన్ని ఇవ్వండి.",
                    "url": "Official India Code URL (if available)."
                }
            }
        ],
        "actions_en": [
            "Step 1 in English",
            "Step 2 in English"
        ],
        "actions_te": [
            "Step 1 in Telugu",
            "Step 2 in Telugu"
        ]
    }

    prompt = f"""
    You are an expert AI legal assistant for Indian Police officers. Your task is to analyze case details (FIR, theft report, victim statement, etc.) and provide a structured, actionable analysis. Do not include any personal or sensitive data from the input.

    Clarity and Detail Requirements:
    - Provide clear, specific, and concise rationale while ensuring useful detail (avoid vagueness).
    - Use exact, verbatim quotes from the provided law text for the reasoning_quote fields.
    - Ensure actions are concrete, operational steps that an officer can follow.

    Analysis Task:
    1. Analyze the FIR Summary carefully.
    2. Recommend the most applicable IPC/CrPC sections.
    3. Provide rationale (2–3 sentences) for each section in both English and Telugu.
    4. Include `reasoning_quote_en` and `reasoning_quote_te` as verbatim quotes from the provided law text.
    5. Suggest 3–5 concrete procedural steps (in English and Telugu).
    6. Cite the full law text and titles in both languages when available.
    7. If Telugu law text or Telugu quotes are unavailable in the provided context, produce a faithful Telugu translation for the Telugu fields (reasoning_quote_te, fir_quote_te, title_te, full_text_te). Do not leave them blank or copy English.
    8. For each recommended section, include an exact FIR excerpt (fir_quote_en, fir_quote_te) copied verbatim from the FIR summary that motivated the recommendation.

    Context from Vector Database:

    Potentially Relevant Law Sections:
    {laws_context}

    Jurisdiction for this case: {jurisdiction}

    Case Details to Analyze:
    ```
    {fir_summary}
    ```

    IMPORTANT: Your final output MUST be a single, valid JSON object. Do not add any text or explanation outside of the JSON structure. The JSON object must conform EXACTLY to the following schema (including the reasoning_quote_* fields):
    ```json
    {json.dumps(json_schema, indent=2)}
    ```
    """
    return prompt

def call_gemini(system_prompt):
    """Calls the Gemini API with the system prompt and returns the JSON response."""
    response_text = ""
    try:
        model = genai.GenerativeModel(
            LLM_MODEL_NAME,
            generation_config={
                "response_mime_type": "application/json",
                "temperature": LLM_TEMPERATURE,
                "max_output_tokens": MAX_OUTPUT_TOKENS,
            }
        )
        response = model.generate_content(system_prompt)
        response_text = getattr(response, "text", "")
        return json.loads(response_text)
    except Exception as e:
        st.error(f"An error occurred with the Gemini API call: {e}")
        try:
            clean_text = (response_text or "").strip().replace("```json", "").replace("```", "")
            if clean_text:
                return json.loads(clean_text)
        except Exception:
            pass
        if response_text:
            st.code(response_text, language="text")
        return None


# --- STREAMLIT UI LAYOUT ---
st.title("⚖️ Police Case Analysis Assistant")
st.warning(
    "**Disclaimer:** This tool provides AI-generated recommendations for internal police use only. "
    "All outputs must be verified by a qualified officer. Do not include sensitive PII in the input. "
    "You can paste FIRs, theft reports, victim statements, or other case details.",
    icon="⚠️"
)
st.info("Paste FIRs, theft reports, victim statements, or any case details into the chat below to analyze.")


# --- CHAT-LIKE MAIN CONTENT ---
def _fallback_te(te_val: str, en_val: str) -> str:
    bads = {
        "", None,
        "తెలుగు పాఠ్యం సందర్భంలో అందించబడలేదు.",
        "తెలుగు అనువాదం అందుబాటులో లేదు",
        "N/A"
    }
    te = (te_val or "").strip()
    if te and te not in bads:
        return te
    en = (en_val or "").strip()
    return translate_to_te(en) if en else ""

def _render_analysis_result(res: dict):
    if not res:
        st.info("No analysis available.")
        return
    st.subheader("✅ Recommended Sections")
    if res.get("sections"):
        for sec in res["sections"]:
            section_label = sec.get("section", "N/A")
            law = sec.get("law_citation", {}) or {}
            title_en = law.get("title_en", "N/A")
            title_te = _fallback_te(law.get("title_te"), title_en)

            st.markdown(f"<div class='card'><strong>Section:</strong> <code>{section_label}</code>", unsafe_allow_html=True)

            col_en, col_te = st.columns(2)
            with col_en:
                st.markdown("**English**")
                st.markdown(f"- Title: {title_en}")
                st.info(f"Rationale: {sec.get('rationale_en', 'N/A')}")
                quote_en = (sec.get("reasoning_quote_en") or "").strip()
                if quote_en:
                    st.markdown("Exact law quote (EN)")
                    st.markdown(f"<div class='quote-block'>{quote_en}</div>", unsafe_allow_html=True)
            with col_te:
                st.markdown("**తెలుగు**")
                st.markdown(f"- శీర్షిక: {title_te or 'N/A'}")
                rationale_te = _fallback_te(sec.get('rationale_te'), sec.get('rationale_en'))
                st.info(f"తర్కం: {rationale_te or 'N/A'}")
                quote_te_raw = (sec.get("reasoning_quote_te") or "").strip()
                quote_te = _fallback_te(quote_te_raw, quote_en)
                if quote_te:
                    st.markdown("యథాతధ చట్ట కోట్ (TE)")
                    st.markdown(f"<div class='quote-block'>{quote_te}</div>", unsafe_allow_html=True)

            with st.expander("View full law text and source"):
                full_en = law.get("full_text_en") or law.get("full_text") or ""
                full_te = _fallback_te(law.get("full_text_te"), full_en)
                if full_en:
                    st.markdown("**Full text (English)**")
                    st.code(full_en, language="text")
                if full_te:
                    st.markdown("**పూర్తి పాఠ్యం (తెలుగు)**")
                    st.code(full_te, language="text")
                if law.get("url"):
                    st.caption(f"Source: {law.get('url')}")

            fir_en = (sec.get("fir_quote_en") or "").strip()
            fir_te = _fallback_te((sec.get("fir_quote_te") or "").strip(), fir_en)
            if fir_en or fir_te:
                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                st.markdown("**FIR excerpt supporting this recommendation**")
                col_fir_en, col_fir_te = st.columns(2)
                with col_fir_en:
                    if fir_en:
                        st.markdown("FIR excerpt (EN)")
                        st.markdown(f"<div class='quote-block'>{fir_en}</div>", unsafe_allow_html=True)
                with col_fir_te:
                    if fir_te:
                        st.markdown("FIR ఉద్దరణ (TE)")
                        st.markdown(f"<div class='quote-block'>{fir_te}</div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    else:
        st.info("No legal sections were recommended.")

    st.subheader("📋 Suggested Procedural Actions")
    col_en_a, col_te_a = st.columns(2)
    with col_en_a:
        st.markdown("**English**")
        for i, action in enumerate(res.get("actions_en", []), 1):
            st.markdown(f"{i}. {action}")
    with col_te_a:
        st.markdown("**తెలుగు**")
        actions_en = res.get("actions_en", [])
        actions_te = res.get("actions_te", [])
        if not actions_te and actions_en:
            actions_te = [translate_to_te(a) for a in actions_en]
        for i, action in enumerate(actions_te, 1):
            st.markdown(f"{i}. {action}")


for msg in st.session_state.messages:
    if msg.get("role") == "user":
        with st.chat_message("user"):
            st.markdown(msg.get("content", ""))
    elif msg.get("role") == "assistant":
        with st.chat_message("assistant"):
            _render_analysis_result(msg.get("analysis_result", {}))

user_input = st.chat_input("Enter case details to analyze…")
if user_input:
    st.session_state.fir_text = user_input
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("🧠 Analyzing... This may take a moment."):
        relevant_laws = retrieve_law_sections(user_input, k=st.session_state.top_k_laws)
        system_prompt = build_system_prompt(user_input, relevant_laws, st.session_state.jurisdiction)
        analysis_result = call_gemini(system_prompt)

    with st.chat_message("assistant"):
        _render_analysis_result(analysis_result or {})

    st.session_state.messages.append({"role": "assistant", "analysis_result": analysis_result or {}})