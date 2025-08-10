
import os
import hashlib
import re
import io
import time
from typing import List, Tuple

from dotenv import load_dotenv
import fitz  # PyMuPDF
from PIL import Image
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import AzureOpenAI

def sanitize_key(key: str) -> str:
    """Convert a string into a valid Azure Search document key."""
    return re.sub(r'[^a-zA-Z0-9_\-=]', '_', key)

# Load environment variables
load_dotenv()

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# Constants
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 400))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))

def hash_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def normalize_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_text_from_image(image_bytes) -> str:
    """Extract text from an image using Azure Computer Vision OCR."""
    try:
        from azure.cognitiveservices.vision.computervision import ComputerVisionClient
        from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
        from msrest.authentication import CognitiveServicesCredentials

        endpoint = os.getenv("AZURE_VISION_ENDPOINT")
        key = os.getenv("AZURE_VISION_KEY")

        vision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(key))
        image_stream = io.BytesIO(image_bytes)

        read_response = vision_client.read_in_stream(image_stream, raw=True)
        operation_location = read_response.headers["Operation-Location"]
        operation_id = operation_location.split("/")[-1]

        # Poll
        while True:
            read_result = vision_client.get_read_result(operation_id)
            if read_result.status not in ['notStarted', 'running']:
                break
            time.sleep(1)

        text = ""
        if read_result.status == OperationStatusCodes.succeeded:
            for text_result in read_result.analyze_result.read_results:
                for line in text_result.lines:
                    text += line.text + "\n"
        return text
    except Exception as e:
        print(f"OCR error: {str(e)}")
        return ""

def extract_tables_from_page(page):
    tables = []
    try:
        tab = page.find_tables()
        if tab.tables:
            for table in tab.tables:
                try:
                    df = pd.DataFrame(table.extract())
                    table_str = df.to_string(index=False, header=False)
                    tables.append(table_str)
                except Exception as e:
                    print(f"Table extraction error: {str(e)}")
    except Exception:
        pass
    return tables

def process_pdf(file_path: str, file_name: str):
    """Process a PDF file to extract text, OCR images if needed, extract tables, then chunk."""
    document_text = []
    pdf_document = fitz.open(file_path)

    for page_num, page in enumerate(pdf_document):
        page_text = page.get_text() or ""

        # If page has little text, try OCR
        if len(page_text.strip()) < 20:
            try:
                pix = page.get_pixmap()
                img_bytes = pix.tobytes()
                ocr_text = extract_text_from_image(img_bytes)
                if ocr_text.strip():
                    document_text.append(f"[Page {page_num+1} OCR]:\n{ocr_text}")
            except Exception as e:
                print(f"OCR render error p{page_num+1}: {e}")

        # Add page text
        if page_text.strip():
            document_text.append(f"[Page {page_num+1} Text]:\n{page_text}")

        # Tables
        tables = extract_tables_from_page(page)
        if tables:
            document_text.append(f"[Page {page_num+1} Table]:\n" + "\n\n".join(tables))

    pdf_document.close()

    combined_text = "\n\n".join(document_text)
    normalized_text = normalize_text(combined_text)
    return chunk_text(normalized_text)

def chunk_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_text(text)
    return [normalize_text(c) for c in chunks if c and c.strip()]

# ---------- Embeddings ----------

def _get_deployment_name() -> str:
    name = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    if not name:
        raise ValueError("Set AZURE_OPENAI_EMBEDDING_DEPLOYMENT in .env (e.g., text-embedding-3-small)")
    return name

def create_embedding(text: str):
    try:
        dep = _get_deployment_name()
        resp = client.embeddings.create(model=dep, input=text)
        emb = resp.data[0].embedding
        return [float(x) for x in emb]
    except Exception as e:
        import traceback
        print(f"Error creating embedding: {e}\n{traceback.format_exc()}")
        # Return zero-vector to keep pipeline moving
        return [0.0] * 1536

def create_embeddings_batch(text_list: List[str]) -> List[List[float]]:
    try:
        dep = _get_deployment_name()
        resp = client.embeddings.create(model=dep, input=text_list)
        return [list(map(float, r.embedding)) for r in resp.data]
    except Exception as e:
        import traceback
        print(f"Error creating batch embeddings: {e}\n{traceback.format_exc()}")
        return [[0.0]*1536 for _ in text_list]

def test_embedding_connection() -> Tuple[bool, str]:
    try:
        dep = _get_deployment_name()
        resp = client.embeddings.create(model=dep, input="hello")
        ok = bool(resp and resp.data and resp.data[0].embedding)
        return (ok, f"Connected using deployment '{dep}'")
    except Exception as e:
        return (False, f"{type(e).__name__}: {e}")
