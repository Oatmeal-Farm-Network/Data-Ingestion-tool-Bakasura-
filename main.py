import os
import time
import streamlit as st
from dotenv import load_dotenv
import embedding_utils as eu
import db_utils as db
from pathlib import Path
import tempfile
from PIL import Image

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Knowledge Devourer | Document Ingestion",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check for required environment variables
required_vars = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "COSMOS_ENDPOINT", "COSMOS_KEY", "AZURE_VISION_ENDPOINT"]
missing_vars = []

for var in required_vars:
    if not os.getenv(var):
        missing_vars.append(var)

if missing_vars:
    st.error(f"⚠️ Missing required environment variables: {', '.join(missing_vars)}")
    st.info("Please check your .env file and ensure all required variables are set.")
    st.stop()  # Stop execution if missing critical environment variables

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        color: #8B0000;
        margin-bottom: 0;
    }
    .sub-title {
        font-size: 1.5rem;
        color: #555;
        margin-top: 0;
    }
    .description {
        color: #666;
        font-style: italic;
        padding: 1rem 0;
    }
    .section-header {
        background-color: #f0f0f0;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #f8f9fa;
        border-left: 5px solid #8B0000;
        padding: 1rem;
        border-radius: 0 5px 5px 0;
        margin: 1rem 0;
    }
    .file-uploader {
        background-color: #f9f9f9;
        padding: 1rem;
        border-radius: 5px;
        border: 1px dashed #aaa;
    }
    .stButton button {
        background-color: #8B0000;
        color: white;
        font-weight: bold;
    }
    .progress-container {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 5px;
        margin-top: 1rem;
    }
    .footer {
        margin-top: 3rem;
        text-align: center;
        font-size: 0.8rem;
        color: #666;
    }
    .stProgress .st-ey {
        background-color: #8B0000;
    }
    .success-msg {
        color: #28a745;
        font-weight: bold;
    }
    .error-msg {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Dashboard Layout
col1, col2 = st.columns([1, 3])

with col1:
    # Image path
    image_path = "/Users/gugullasneha/Desktop/images/bakasura.jpeg"  # or .jpg depending on your file
    
    if os.path.exists(image_path):
        image = Image.open(image_path)
        st.image(image, width=200)
    else:
        st.warning(f"Image not found at {image_path}. Please add an image named 'bakasura.png' to the images folder.")

with col2:
    st.markdown('<p class="main-title">Bakasura Knowledge Devourer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Document Ingestion System</p>', unsafe_allow_html=True)
    st.markdown('<p class="description">Upload your PDFs and feed the insatiable appetite of our knowledge system.</p>', unsafe_allow_html=True)

# Sidebar content
with st.sidebar:
    st.title("About the System")
    st.markdown("""
    ### What is Bakasura?
    In Hindu mythology, Bakasura was a demon known for his insatiable hunger. 
    
    Just like Bakasura, our system devours documents and processes their knowledge for later retrieval.
    """)
    
    st.markdown("""
    ### System Capabilities
    - Extract text from PDFs
    - Process tables and images with OCR
    - Create semantic embeddings
    - Store in vector database
    """)
    
    st.markdown("---")
    
    # Add statistics if you can fetch them from the database
    try:
        # If you have a way to get stats, replace these with actual values
        total_docs = "N/A"  # Replace with actual count
        total_chunks = "N/A"  # Replace with actual count
        
        st.subheader("System Statistics")
        st.metric("Documents Processed", total_docs)
        st.metric("Knowledge Chunks", total_chunks)
    except:
        st.info("Statistics coming soon")
        
    # Test Azure OpenAI Connection
    st.markdown("---")
    st.subheader("Diagnostics")
    if st.button("Test Azure OpenAI Connection"):
        try:
            with st.spinner("Testing connection to Azure OpenAI..."):
                test_text = "This is a test sentence."
                embedding = eu.create_embedding(test_text)
                if any(embedding):  # Check if embedding is not all zeros
                    st.success("✅ Successfully connected to Azure OpenAI!")
                    st.write(f"Embedding dimensions: {len(embedding)}")
                else:
                    st.error("⚠️ Got a zero embedding. Check your Azure OpenAI configuration.")
        except Exception as e:
            st.error(f"❌ Error connecting to Azure OpenAI: {str(e)}")

# Main content
st.markdown('<div class="section-header"><h3>📄 Document Upload</h3></div>', unsafe_allow_html=True)

st.markdown('<div class="info-box">Our system can process PDFs containing text, tables, and images. All content will be extracted, analyzed, and stored for retrieval.</div>', unsafe_allow_html=True)

# Initialize session state for tracking progress
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'progress' not in st.session_state:
    st.session_state.progress = {}
if 'total_chunks' not in st.session_state:
    st.session_state.total_chunks = 0

# File uploader with improved styling
st.markdown('<div class="file-uploader">', unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    "Select PDF files to process (max 10)", 
    type=["pdf"], 
    accept_multiple_files=True,
    disabled=st.session_state.processing
)
st.markdown('</div>', unsafe_allow_html=True)

# Process button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    process_btn = st.button(
        "🔄 Process Documents", 
        disabled=st.session_state.processing or not uploaded_files,
        key="process_button"
    )

# Progress display area
progress_container = st.empty()
status_container = st.empty()

# Initialize database connection
db_client = None
try:
    db_client = db.initialize_cosmos_client()
    status_container.success("✅ Connected to Azure Cosmos DB")
except Exception as e:
    status_container.error(f"❌ Error connecting to Cosmos DB: {str(e)}")

# Process documents when button is clicked
if process_btn and uploaded_files and db_client:
    st.session_state.processing = True
    st.session_state.progress = {}
    st.session_state.total_chunks = 0
    
    # Create a progress container
    with progress_container.container():
        st.markdown('<div class="progress-container">', unsafe_allow_html=True)
        st.subheader("Processing Documents")
        
        # Display overall progress bar
        progress_bar = st.progress(0)
        
        # Limit number of files
        max_files = int(os.getenv("MAX_FILES", 10))
        if len(uploaded_files) > max_files:
            st.warning(f"⚠️ Only the first {max_files} files will be processed.")
            uploaded_files = uploaded_files[:max_files]
        
        # Process each file
        for i, uploaded_file in enumerate(uploaded_files):
            file_name = uploaded_file.name
            st.session_state.progress[file_name] = {"status": "Processing...", "chunks": 0}
            
            # Update progress information
            file_progress_text = st.empty()
            file_progress_text.info(f"Processing {file_name}...")
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name
            
            try:
                # Extract text (with OCR if needed)
                with st.spinner(f"Extracting text from {file_name}..."):
                    text_chunks = eu.process_pdf(temp_path, file_name)
                    file_progress_text.info(f"Extracted {len(text_chunks)} text chunks from {file_name}. Creating embeddings...")
                
                # Create embeddings and store in DB
                chunk_count = 0
                for chunk_id, chunk_text in enumerate(text_chunks):
                    # Show chunk progress
                    progress_bar.progress((i + (chunk_id + 1) / len(text_chunks)) / len(uploaded_files))
                    
                    # Generate embedding
                    embedding = eu.create_embedding(chunk_text)
                    
                    # Create metadata
                    metadata = {
                        "filename": file_name,
                        "chunk_id": chunk_id,
                        "timestamp": time.time(),
                        "text_hash": eu.hash_text(chunk_text)
                    }
                    
                    # Store in Cosmos DB
                    success = db.store_embedding(db_client, chunk_text, embedding, metadata)
                    
                    if success:
                        chunk_count += 1
                
                # Update progress
                st.session_state.progress[file_name] = {
                    "status": "Completed", 
                    "chunks": chunk_count
                }
                st.session_state.total_chunks += chunk_count
                
                # Show completion message for this file
                file_progress_text.success(f"✅ {file_name}: Processed {chunk_count} chunks successfully")
                
            except Exception as e:
                st.session_state.progress[file_name] = {"status": f"Error: {str(e)}", "chunks": 0}
                file_progress_text.error(f"❌ Error processing {file_name}: {str(e)}")
                
            finally:
                # Clean up the temporary file
                os.unlink(temp_path)
        
        # Processing complete
        st.session_state.processing = False
        
        # Show final summary
        if st.session_state.total_chunks > 0:
            st.markdown(f'<p class="success-msg">🎉 Success! {st.session_state.total_chunks} chunks ingested from {len(uploaded_files)} documents.</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="error-msg">⚠️ No chunks were ingested. Please check the errors above.</p>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Display an explainer section when not processing
if not st.session_state.processing:
    st.markdown('<div class="section-header"><h3>📚 How It Works</h3></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 1. Upload")
        st.markdown("""
        - Upload PDF documents
        - Supports text, tables, images
        - Multi-file upload supported
        """)
    
    with col2:
        st.markdown("### 2. Process")
        st.markdown("""
        - Text extraction from PDFs
        - OCR for images using Azure
        - Table detection and processing
        - Text chunking and normalization
        """)
    
    with col3:
        st.markdown("### 3. Store")
        st.markdown("""
        - Generate embeddings with Azure OpenAI
        - Store in Azure Cosmos DB
        - Deduplication of content
        - Ready for retrieval
        """)

# Footer
st.markdown('<div class="footer">© 2025 Your Organization Name • Built with Azure, OpenAI, and Streamlit</div>', unsafe_allow_html=True)