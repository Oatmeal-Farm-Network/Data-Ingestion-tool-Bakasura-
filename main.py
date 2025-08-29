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


# --- Function to get vector index statistics ---
def get_index_stats(search_client, index_name):
    """Get statistics about the vector index"""
    try:
        # Get document count from the index
        search_results = search_client.search(search_text="*", top=1, include_total_count=True)
        total_docs = search_results.get_count()
        
        # You can add more statistics here as needed
        return {
            "total_documents": total_docs,
            "index_name": index_name,
            "status": "connected"
        }
    except Exception as e:
        return {
            "total_documents": 0,
            "index_name": index_name,
            "status": "error",
            "error": str(e)
        }


# --- Function to display processing progress ---
def display_processing_progress(file_name, current_chunk, total_chunks, status="processing"):
    """Display real-time processing progress for each file"""
    progress_percentage = (current_chunk / total_chunks) * 100 if total_chunks > 0 else 0
    
    if status == "processing":
        st.write(f"🔄 **{file_name}** - Chunk {current_chunk}/{total_chunks} ({progress_percentage:.1f}%)")
        st.progress(progress_percentage / 100)
    elif status == "success":
        st.write(f"✅ **{file_name}** - Successfully processed {total_chunks} chunks")
        st.progress(1.0)
    elif status == "error":
        st.write(f"❌ **{file_name}** - Failed processing")
        st.progress(0.0)


# Load Custom CSS
load_css("styles/main.css")

# --- Environment Variable Check ---
required_vars = [
    "azure-openai-api-key",
    "azure-openai-endpoint",
    "azure-vision-endpoint",
    "azure-vision-key",
    "azure-search-endpoint",
    "azure-search-key",
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
    
    # Test Azure OpenAI Connection
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
    
    # Test Azure AI Search Connection and show index stats
    if st.button("Test Azure AI Search & Get Index Info"):
        try:
            with st.spinner("Testing Azure AI Search..."):
                search_client, index_client = db.initialize_search_client()
                
                # Get the index name (you might need to modify this based on your db_utils implementation)
                index_name = getattr(search_client, '_index_name', 'documents')  # fallback name
                
                index_stats = get_index_stats(search_client, index_name)
                
                if index_stats["status"] == "connected":
                    st.success("✅ Connected to Azure AI Search")
                    st.info(f"📊 Index: **{index_stats['index_name']}**")
                    st.info(f"📄 Total Documents: **{index_stats['total_documents']}**")
                else:
                    st.error(f"❌ Failed to get index stats: {index_stats.get('error', 'Unknown error')}")
        except Exception as e:
            st.error(f"❌ Failed to connect to Azure AI Search: {e}")

# --- Vector Index Information Section ---
st.markdown(
    '<div class="section-header"><h3>🔍 Vector Index Information</h3></div>',
    unsafe_allow_html=True,
)

# Initialize session state for index stats
if "index_stats" not in st.session_state:
    st.session_state.index_stats = None

# Auto-load index stats on page load
if st.session_state.index_stats is None:
    try:
        search_client, index_client = db.initialize_search_client()
        index_name = getattr(search_client, '_index_name', 'documents')
        st.session_state.index_stats = get_index_stats(search_client, index_name)
    except Exception as e:
        st.session_state.index_stats = {
            "total_documents": 0,
            "index_name": "unknown",
            "status": "error",
            "error": str(e)
        }

# Display current index stats
if st.session_state.index_stats:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.index_stats["status"] == "connected":
            st.metric("Connection Status", "🟢 Connected")
        else:
            st.metric("Connection Status", "🔴 Disconnected")
    
    with col2:
        st.metric("Index Name", st.session_state.index_stats["index_name"])
    
    with col3:
        st.metric("Total Documents", st.session_state.index_stats["total_documents"])
    
    if st.session_state.index_stats["status"] == "error":
        st.error(f"⚠️ Index Error: {st.session_state.index_stats.get('error', 'Unknown error')}")

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

if "processing_stats" not in st.session_state:
    st.session_state.processing_stats = {
        "total_files": 0,
        "successful_files": 0,
        "failed_files": 0,
        "total_chunks": 0,
        "failed_chunks": 0
    }

uploaded_files = st.file_uploader(
    "Select PDF files (up to 10)", type=["pdf"], accept_multiple_files=True
)

if st.button("✨ Process Documents", disabled=(not uploaded_files)):
    st.session_state.processing_complete = False
    st.session_state.processing_stats = {
        "total_files": len(uploaded_files),
        "successful_files": 0,
        "failed_files": 0,
        "total_chunks": 0,
        "failed_chunks": 0
    }
    
    try:
        search_client, index_client = db.initialize_search_client()
        index_name = getattr(search_client, '_index_name', 'documents')
        st.success(f"✅ Connected to Azure AI Search - Index: **{index_name}**")
        
        # Get initial document count
        initial_stats = get_index_stats(search_client, index_name)
        initial_doc_count = initial_stats["total_documents"]
        st.info(f"📊 Initial document count: **{initial_doc_count}**")
        
    except Exception as e:
        st.error(f"❌ Failed to connect to Azure AI Search: {e}")
        st.stop()

    # Create containers for real-time updates
    progress_container = st.container()
    stats_container = st.container()
    error_container = st.container()
    
    with st.status("Processing Documents...") as status:
        total_chunks_ingested = 0
        error_logs = []
        
        for file_idx, uploaded_file in enumerate(uploaded_files):
            file_name = uploaded_file.name
            
            with progress_container:
                st.markdown(f"### Processing File {file_idx + 1}/{len(uploaded_files)}")
                file_progress_placeholder = st.empty()
                chunk_progress_placeholder = st.empty()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name
            
            try:
                # Extract text chunks
                with file_progress_placeholder:
                    st.write(f"🔍 **{file_name}** - Extracting text chunks...")
                
                text_chunks = eu.process_pdf(temp_path, file_name)
                total_file_chunks = len(text_chunks)
                
                with file_progress_placeholder:
                    st.write(f"📄 **{file_name}** - Found {total_file_chunks} chunks. Starting vectorization...")
                
                successful_chunks = 0
                failed_chunks_in_file = 0
                
                # Process each chunk
                for i, chunk_text in enumerate(text_chunks):
                    try:
                        # Update progress for current chunk
                        with chunk_progress_placeholder:
                            display_processing_progress(file_name, i + 1, total_file_chunks, "processing")
                        
                        # Create embedding
                        embedding = eu.create_embedding(chunk_text)
                        
                        # Prepare metadata
                        metadata = {
                            "filename": file_name,
                            "chunk_id": i,
                            "timestamp": time.time(),
                            "text_hash": eu.hash_text(chunk_text),
                            "page_number": i + 1,
                        }
                        
                        # Store in vector database
                        doc_key = sanitize_key(f"{file_name}_{i}")
                        try:
                            # Try using improved store_embedding if available
                            if hasattr(db, 'store_embedding') and 'store_embedding_legacy' in dir(db):
                                success, error_msg = db.store_embedding(
                                    search_client, chunk_text, embedding, metadata, doc_key
                                )
                                if success:
                                    successful_chunks += 1
                                    total_chunks_ingested += 1
                                else:
                                    failed_chunks_in_file += 1
                                    error_logs.append(f"Chunk {i} from {file_name}: {error_msg or 'Unknown error'}")
                            else:
                                # Fallback to original function
                                success = db.store_embedding(
                                    search_client, chunk_text, embedding, metadata, doc_key
                                )
                                
                                if success:
                                    successful_chunks += 1
                                    total_chunks_ingested += 1
                                else:
                                    failed_chunks_in_file += 1
                                    # Check if it's a duplicate content issue
                                    text_hash = metadata.get("text_hash")
                                    if text_hash:
                                        error_logs.append(f"Chunk {i} from {file_name} - Likely duplicate content (text_hash: {text_hash[:10]}...)")
                                    else:
                                        error_logs.append(f"Chunk {i} from {file_name} - store_embedding returned False (unknown reason)")
                                        
                        except Exception as store_error:
                            failed_chunks_in_file += 1
                            error_logs.append(f"Chunk {i} from {file_name} - Exception: {str(store_error)}")
                            
                    except Exception as chunk_error:
                        failed_chunks_in_file += 1
                        error_logs.append(f"Error processing chunk {i} from {file_name}: {str(chunk_error)}")
                
                # Final status for this file
                with chunk_progress_placeholder:
                    if failed_chunks_in_file == 0:
                        display_processing_progress(file_name, total_file_chunks, total_file_chunks, "success")
                        st.session_state.processing_stats["successful_files"] += 1
                    else:
                        st.write(f"⚠️ **{file_name}** - Completed with {failed_chunks_in_file} failed chunks")
                        st.session_state.processing_stats["failed_files"] += 1
                
                st.session_state.processing_stats["total_chunks"] += successful_chunks
                st.session_state.processing_stats["failed_chunks"] += failed_chunks_in_file
                
            except Exception as file_error:
                error_logs.append(f"Critical error processing {file_name}: {str(file_error)}")
                st.session_state.processing_stats["failed_files"] += 1
                
                with chunk_progress_placeholder:
                    display_processing_progress(file_name, 0, 1, "error")
                    st.error(f"❌ Failed to process {file_name}: {str(file_error)}")
            
            finally:
                os.unlink(temp_path)
        
        # Display final statistics
        with stats_container:
            st.markdown("### 📊 Processing Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Files Processed", 
                         f"{st.session_state.processing_stats['successful_files']}/{st.session_state.processing_stats['total_files']}")
            with col2:
                st.metric("Chunks Ingested", st.session_state.processing_stats['total_chunks'])
            with col3:
                st.metric("Failed Chunks", st.session_state.processing_stats['failed_chunks'])
            with col4:
                # Get updated document count
                try:
                    updated_stats = get_index_stats(search_client, index_name)
                    new_doc_count = updated_stats["total_documents"]
                    st.metric("Total Index Documents", new_doc_count, delta=new_doc_count - initial_doc_count)
                    
                    # Update session state with new stats
                    st.session_state.index_stats = updated_stats
                except:
                    st.metric("Total Index Documents", "Error")
        
        # Display errors if any
        if error_logs:
            with error_container:
                st.markdown("### ⚠️ Error Details")
                with st.expander(f"Show {len(error_logs)} errors"):
                    for error in error_logs:
                        st.error(error)

    status.update(
        label=f"Processing complete! Ingested {total_chunks_ingested} chunks from {st.session_state.processing_stats['successful_files']} files.",
        state="complete" if st.session_state.processing_stats['failed_files'] == 0 else "error",
    )
    st.session_state.processing_complete = True

if st.session_state.processing_complete:
    success_rate = (st.session_state.processing_stats['successful_files'] / 
                   st.session_state.processing_stats['total_files'] * 100) if st.session_state.processing_stats['total_files'] > 0 else 0
    
    if success_rate == 100:
        st.success("🎉 All files processed successfully and stored in Azure AI Search!")
    elif success_rate > 0:
        st.warning(f"⚠️ Processing completed with {success_rate:.1f}% success rate. Check error details above.")
    else:
        st.error("❌ All files failed to process. Please check the error details above.")

st.markdown("---")
st.markdown(
    '<div class="footer">© 2025 Bakasura Project • Built with Azure, OpenAI, and Streamlit</div>',
    unsafe_allow_html=True,
)