
import os
import time
import json
import datetime as dt
import streamlit as st
from dotenv import load_dotenv
import embedding_utils as eu
import db_utils as db
import tempfile
from PIL import Image
from embedding_utils import sanitize_key

# Load environment variables
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="Bakasura Knowledge Devourer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Function to load and inject CSS ---
def load_css(file_path):
    try:
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

# Load Custom CSS (optional)
load_css("styles/main.css")

# --- Environment Variable Check ---
required_vars = [
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_VERSION",
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
        st.image(image, width=180)
    except Exception:
        st.empty()

with col2:
    st.markdown('<h1 class="main-title">Bakasura Knowledge Devourer</h1>', unsafe_allow_html=True)
    st.caption("High-throughput PDF ingestion → Azure OpenAI embeddings → Azure AI Search vectors")

# --- Sidebar ---
with st.sidebar:
    st.title("Diagnostics")
    st.code(
        f"Deployment: {os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT', '❌ not set')}\n"
        f"API Version: {os.getenv('AZURE_OPENAI_API_VERSION', '❌ not set')}\n"
        f"Endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT', '❌ not set')}"
    )
    if st.button("Test Azure OpenAI Connection"):
        try:
            with st.spinner("Testing Azure OpenAI embeddings..."):
                ok, msg = eu.test_embedding_connection()
                if ok:
                    st.success(f"✅ {msg}")
                else:
                    st.error(f"❌ {msg}")
        except Exception as e:
            st.error(f"❌ Failed: {e}")

# --- Main Section ---
st.subheader("📄 Document Upload")
st.info("Upload one or more PDFs. The app will parse → chunk → embed in batches → upload to Azure AI Search in bulk.")

if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

uploaded_files = st.file_uploader(
    "Select PDF files (up to 10)",
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("✨ Process Documents", disabled=(not uploaded_files)):
    st.session_state.processing_complete = False
    try:
        search_client, index_client = db.initialize_search_client()
        st.success("✅ Connected to Azure AI Search")
    except Exception as e:
        st.error(f"❌ Failed to connect to Azure AI Search: {e}")
        st.stop()

    with st.status("Processing Documents...", expanded=True) as status:
        total_chunks_ingested = 0
        EMBED_BATCH = int(os.getenv("EMBED_BATCH", 32))
        UPLOAD_BATCH = int(os.getenv("UPLOAD_BATCH", 100))

        file_progress = st.progress(0, text="📁 Files")
        files_done = 0
        total_files = len(uploaded_files)

        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            st.write(f"**Processing:** {file_name}")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name

            try:
                t0 = dt.datetime.now()
                text_chunks = eu.process_pdf(temp_path, file_name)
                st.write(f"→ Found **{len(text_chunks)}** chunks "
                         f"(parse time: {(dt.datetime.now()-t0).total_seconds():.1f}s).")

                # Chunk progress
                chunk_progress = st.progress(0, text="📄 Embedding & Uploading")
                pending_docs = []

                e0 = dt.datetime.now()
                for i in range(0, len(text_chunks), EMBED_BATCH):
                    chunk_batch = text_chunks[i:i + EMBED_BATCH]

                    # Skip empty chunks silently
                    chunk_batch = [c for c in chunk_batch if c and c.strip()]
                    if not chunk_batch:
                        continue

                    embeddings = eu.create_embeddings_batch(chunk_batch)

                    # Build docs for bulk upload
                    for j, (chunk_text, embedding) in enumerate(zip(chunk_batch, embeddings)):
                        global_chunk_id = i + j
                        metadata = {
                            "filename": file_name,
                            "chunk_id": global_chunk_id,
                            "timestamp": time.time(),
                            "text_hash": eu.hash_text(chunk_text),
                            "page_number": global_chunk_id + 1
                        }
                        doc_key = sanitize_key(f"{file_name}_{global_chunk_id}")
                        pending_docs.append(db.document_from_parts(doc_key, chunk_text, embedding, metadata))

                        # Flush to search in batches
                        if len(pending_docs) >= UPLOAD_BATCH:
                            ing = db.store_embeddings_bulk(search_client, pending_docs, batch_size=UPLOAD_BATCH)
                            total_chunks_ingested += ing
                            pending_docs = []

                    # update progress
                    chunk_progress.progress(min((i + EMBED_BATCH) / max(len(text_chunks), 1), 1.0),
                                            text=f"📄 {min(i+EMBED_BATCH, len(text_chunks))}/{len(text_chunks)}")

                # Flush leftovers
                if pending_docs:
                    ing = db.store_embeddings_bulk(search_client, pending_docs, batch_size=UPLOAD_BATCH)
                    total_chunks_ingested += ing

                st.success(f"✅ Stored **{total_chunks_ingested}** chunks so far "
                           f"(embed+upload: {(dt.datetime.now()-e0).total_seconds():.1f}s).")
            except Exception as e:
                st.error(f"Error processing {file_name}: {e}")
            finally:
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass

            files_done += 1
            file_progress.progress(files_done / total_files, text=f"📁 {files_done}/{total_files} files")

    status.update(label=f"✅ Processing complete! Ingested {total_chunks_ingested} chunks.", state="complete")
    st.session_state.processing_complete = True

if st.session_state.processing_complete:
    st.success("🎉 All files processed and stored in Azure AI Search.")

st.markdown("---")
st.caption("© 2025 Bakasura Project • Azure OpenAI + Azure AI Search + Streamlit")
