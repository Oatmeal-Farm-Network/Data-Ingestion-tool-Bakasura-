
import os
import json
import uuid
from datetime import datetime
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticSearch,
    SemanticPrioritizedFields,
    SemanticField
)
from azure.core.credentials import AzureKeyCredential

load_dotenv()

SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "bakasura-documents")
VECTOR_DIMENSIONS = int(os.getenv("VECTOR_DIMENSIONS", "1536"))

def initialize_search_client():
    if not SEARCH_ENDPOINT or not SEARCH_KEY:
        raise ValueError("Azure AI Search endpoint and key must be provided in environment variables")

    credential = AzureKeyCredential(SEARCH_KEY)
    index_client = SearchIndexClient(endpoint=SEARCH_ENDPOINT, credential=credential)
    search_client = SearchClient(endpoint=SEARCH_ENDPOINT, index_name=SEARCH_INDEX_NAME, credential=credential)

    _create_or_update_index(index_client)
    return search_client, index_client

def _create_or_update_index(index_client: SearchIndexClient):
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String, searchable=True),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=VECTOR_DIMENSIONS,
            vector_search_profile_name="my-vector-config"
        ),
        SimpleField(name="filename", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="chunk_id", type=SearchFieldDataType.Int32, filterable=True),
        SimpleField(name="text_hash", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="timestamp", type=SearchFieldDataType.DateTimeOffset, filterable=True, sortable=True),
        SimpleField(name="file_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="page_number", type=SearchFieldDataType.Int32, filterable=True),
        SearchableField(name="metadata", type=SearchFieldDataType.String, searchable=True)
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="my-hnsw-config",
                parameters={"m": 8, "efConstruction": 200, "efSearch": 100, "metric": "cosine"}
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="my-vector-config",
                algorithm_configuration_name="my-hnsw-config"
            )
        ]
    )

    semantic_config = SemanticConfiguration(
        name="my-semantic-config",
        prioritized_fields=SemanticPrioritizedFields(
            content_fields=[SemanticField(field_name="content")]
        )
    )
    semantic_search = SemanticSearch(configurations=[semantic_config])

    index = SearchIndex(
        name=SEARCH_INDEX_NAME,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search
    )

    index_client.create_or_update_index(index)
    print(f"✅ Azure AI Search index '{SEARCH_INDEX_NAME}' is ready")

def document_from_parts(doc_key, text, embedding, metadata: dict):
    return {
        "id": doc_key,
        "content": text,
        "content_vector": embedding,
        "filename": metadata.get("filename"),
        "chunk_id": metadata.get("chunk_id"),
        "text_hash": metadata.get("text_hash"),
        "timestamp": datetime.fromtimestamp(metadata.get("timestamp", 0)).isoformat() + "Z",
        "file_type": "pdf",
        "page_number": metadata.get("page_number"),
        "metadata": json.dumps(metadata)
    }

def store_embeddings_bulk(search_client: SearchClient, documents, batch_size=100, max_retries=3):
    """Upload documents to Azure Search in batches with simple retry/backoff."""
    from time import sleep
    total = 0
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        attempt = 0
        while True:
            try:
                results = search_client.upload_documents(documents=batch)
                total += sum(1 for r in results if getattr(r, "succeeded", False))
                break
            except Exception as e:
                attempt += 1
                if attempt > max_retries:
                    print(f"❌ Bulk upload failed permanently: {e}")
                    break
                sleep(1.5 * attempt)  # backoff
    return total
