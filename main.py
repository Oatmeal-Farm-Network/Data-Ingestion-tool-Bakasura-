import os
import time
import streamlit as st
from dotenv import load_dotenv
import embedding_utils as eu
import db_utils as db
from pathlib import Path
import tempfile
from PIL import Image
from embedding_utils import sanitize_key

# Loading environment variables
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="Bakasura Knowledge Devourer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Function to load and inject CSS ---
def load_css(file_path):
    try:
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"⚠️ CSS file not found at {file_path}.")


# Load Custom CSS
load_css("styles/main.css")

# --- Environment Variable Check ---
required_vars = [
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_VISION_ENDPOINT",
    "AZURE_VISION_KEY",
    "AZURE_SEARCH_ENDPOINT",
    "AZURE_SEARCH_KEY",
]
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    st.error(f"Missing Critical Configuration: {', '.join(missing_vars)}")
    st.stop()

# --- Header Section ---
col1, col2 = st.columns([1, 4])
with col1:
    image_path = "images/bakasura.jpeg"
    try:
        image = Image.open(image_path)
        st.image(image, width=200)
    except FileNotFoundError:
        st.warning(f"Image not found at '{image_path}'")

with col2:
    st.markdown(
        '<h1 class="main-title">Bakasura Knowledge Devourer</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-title">Your central hub for document ingestion and knowledge processing.</p>',
        unsafe_allow_html=True,
    )

# --- Sidebar ---
with st.sidebar:
    st.title("About the System")
    st.markdown(
        """
    Like the mythological Bakasura, this system devours documents to extract and process their knowledge for your use.
    """
    )
    with st.expander("System Capabilities", expanded=True):
        st.markdown(
            """
        - **Text Extraction** from PDFs
        - **OCR** using Azure Vision
        - **Embeddings** using Azure OpenAI
        - **Storage** in Azure AI Search
        """
        )
    st.subheader("System Diagnostics")
    if st.button("Test Azure OpenAI Connection"):
        try:
            with st.spinner("Testing Azure OpenAI..."):
                test = eu.create_embedding("test")
                if any(test):
                    st.success("✅ Connected to Azure OpenAI")
                else:
                    st.warning("Empty embedding returned")
        except Exception as e:
            st.error(f"❌ Failed: {e}")

# --- Main Section ---
st.markdown(
    '<div class="section-header"><h3>📄 Document Upload</h3></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="info-box">Upload PDF files containing text, images, or tables for processing and ingestion.</div>',
    unsafe_allow_html=True,
)

if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False

uploaded_files = st.file_uploader(
    "Select PDF files (up to 10)", type=["pdf"], accept_multiple_files=True
)

if st.button("✨ Process Documents", disabled=(not uploaded_files)):
    st.session_state.processing_complete = False
    try:
        search_client, index_client = db.initialize_search_client()
        st.success("✅ Connected to Azure AI Search")
    except Exception as e:
        st.error(f"❌ Failed to connect to Azure AI Search: {e}")
        st.stop()

    with st.status("Processing Documents...") as status:
        total_chunks_ingested = 0
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            st.write(f"Processing: **{file_name}**")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name
            try:
                text_chunks = eu.process_pdf(temp_path, file_name)
                st.write(f"→ Found {len(text_chunks)} chunks.")
                for i, chunk_text in enumerate(text_chunks):
                    embedding = eu.create_embedding(chunk_text)
                    metadata = {
                        "filename": file_name,
                        "chunk_id": i,
                        "timestamp": time.time(),
                        "text_hash": eu.hash_text(chunk_text),
                        "page_number": i + 1,
                    }
                    doc_key = sanitize_key(f"{file_name}_{i}")
                    success = db.store_embedding(
                        search_client, chunk_text, embedding, metadata, doc_key
                    )
                    if success:
                        total_chunks_ingested += 1
                st.write(f"✅ Stored {total_chunks_ingested} chunks for {file_name}.")
            except Exception as e:
                st.error(f"Error processing {file_name}: {e}")
            finally:
                os.unlink(temp_path)

    status.update(
        label=f"Processing complete! Ingested {total_chunks_ingested} chunks.",
        state="complete",
    )
    st.session_state.processing_complete = True

if st.session_state.processing_complete:
    st.success("🎉 All files processed and stored in Azure AI Search.")

st.markdown("---")
st.markdown(
    '<div class="footer">© 2025 Bakasura Project • Built with Azure, OpenAI, and Streamlit</div>',
    unsafe_allow_html=True,
)
