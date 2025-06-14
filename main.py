import os
import time
import streamlit as st
from dotenv import load_dotenv
import embedding_utils as eu
import db_utils as db
from pathlib import Path
import tempfile
from PIL import Image

# Load environment variables from .env file for local development
load_dotenv()

# --- Page Configuration (Should be the first Streamlit command) ---
st.set_page_config(
    page_title="Bakasura Knowledge Devourer",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Function to load and inject CSS ---
def load_css(file_path):
    """Loads a CSS file and injects it into the Streamlit app."""
    try:
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"⚠️ CSS file not found at {file_path}. Please check the path.")
        st.info("The application will use default Streamlit styling.")

# --- Load Custom CSS ---
# This path is relative to the script's location, which works in Docker.
load_css("styles/main.css")

# --- Environment Variable Check ---
# Checks for variables required for the app to function
required_vars = [
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_VISION_ENDPOINT",
    "COSMOS_ENDPOINT",
    "COSMOS_KEY"
]
missing_vars = [var for var in required_vars if not os.getenv(var)]

if missing_vars:
    st.error(f"**Missing Critical Configuration!** The following environment variables are not set: `{', '.join(missing_vars)}`.")
    st.warning("Please provide these credentials using an `.env` file when running locally, or pass them to the Docker container at runtime.")
    st.info("The application cannot proceed without these settings.")
    st.stop()


# --- Header Section ---
col1, col2 = st.columns([1, 4])

with col1:
    # Correct, relative path for the image that works inside the Docker container
    image_path = "images/bakasura.jpeg"
    try:
        # Using a try-except block is safer for file operations
        image = Image.open(image_path)
        st.image(image, width=200)
    except FileNotFoundError:
        st.warning(f"Image not found at '{image_path}'.")
        st.info("Please ensure 'bakasura.jpeg' is inside the 'images' folder.")

with col2:
    st.markdown('<h1 class="main-title">Bakasura Knowledge Devourer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Your central hub for document ingestion and knowledge processing.</p>', unsafe_allow_html=True)


# --- Sidebar ---
with st.sidebar:
    st.title("About the System")
    st.markdown("""
    Like the mythological Bakasura, known for his insatiable hunger, this system devours documents to extract and process their knowledge for your use.
    """)

    with st.expander("System Capabilities", expanded=True):
        st.markdown("""
        - **Text Extraction:** Pulls text from PDF documents.
        - **OCR Processing:** Extracts text from tables and images using Azure AI Vision.
        - **Semantic Embeddings:** Creates vector representations of your content.
        - **Vector Storage:** Stores and indexes embeddings in Azure Cosmos DB for MongoDB (vCore).
        """)

    st.markdown("---")

    # --- System Diagnostics in Sidebar ---
    st.subheader("System Diagnostics")
    if st.button("Test Azure OpenAI Connection"):
        try:
            with st.spinner("Connecting to Azure OpenAI..."):
                test_embedding = eu.create_embedding("This is a connection test.")
                if test_embedding and any(test_embedding):
                    st.success("Successfully connected to Azure OpenAI!")
                else:
                    st.warning("Connection successful, but received an empty embedding. Check model deployment.")
        except Exception as e:
            st.error(f"Connection Failed: {e}")

# --- Main Content ---
st.markdown('<div class="section-header"><h3>📄 Document Upload</h3></div>', unsafe_allow_html=True)
st.markdown('<div class="info-box">Process PDFs containing text, tables, and images. All content will be extracted, analyzed, and stored for retrieval.</div>', unsafe_allow_html=True)


# --- Session State Initialization ---
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

# --- File Uploader ---
with st.container():
    st.markdown('<div class="file-uploader">', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Select PDF files to process (up to 10)",
        type=["pdf"],
        accept_multiple_files=True
    )
    st.markdown('</div>', unsafe_allow_html=True)


# --- Processing Logic ---
if st.button("✨ Process Documents", disabled=(not uploaded_files)):
    st.session_state.processing_complete = False
    
    # Initialize DB Connection
    try:
        db_client = db.initialize_cosmos_client()
        st.success("✅ Connected to Azure Cosmos DB.")
    except Exception as e:
        st.error(f"❌ Failed to connect to Cosmos DB: {e}")
        st.stop()

    with st.status("Processing Documents...", expanded=True) as status:
        total_chunks_ingested = 0
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            st.write(f"Processing: **{file_name}**")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name

            try:
                # Process the PDF to get text chunks
                text_chunks = eu.process_pdf(temp_path, file_name)
                st.write(f"→ Found {len(text_chunks)} text chunks.")
                
                # Create embeddings and store them
                embeddings_to_store = []
                for i, chunk_text in enumerate(text_chunks):
                    embedding = eu.create_embedding(chunk_text)
                    metadata = {
                        "filename": file_name,
                        "chunk_id": i,
                        "timestamp": time.time(),
                        "text_hash": eu.hash_text(chunk_text)
                    }
                    embeddings_to_store.append((chunk_text, embedding, metadata))
                
                # Store in database
                success_count = db.store_embeddings_batch(db_client, embeddings_to_store)
                total_chunks_ingested += success_count
                st.write(f"→ Successfully stored {success_count} new chunks in the database.")

            except Exception as e:
                st.error(f"An error occurred while processing {file_name}: {e}")
            finally:
                os.unlink(temp_path)
    
    status.update(label=f"Processing Complete! Ingested {total_chunks_ingested} total chunks.", state="complete", expanded=False)
    st.session_state.processing_complete = True


# --- Display results after processing ---
if st.session_state.processing_complete:
    st.success("🎉 All files have been processed. The extracted knowledge is now ready for retrieval.")


# --- Footer ---
st.markdown("---")
st.markdown('<div class="footer">© 2025 Bakasura Project • Built with Azure, OpenAI, and Streamlit</div>', unsafe_allow_html=True)

