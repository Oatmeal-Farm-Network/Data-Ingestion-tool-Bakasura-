import os
import json
from azure.cosmos import CosmosClient, PartitionKey
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT")
COSMOS_KEY = os.getenv("COSMOS_KEY")
COSMOS_DATABASE = os.getenv("COSMOS_DATABASE")
COSMOS_CONTAINER = os.getenv("COSMOS_CONTAINER")


def initialize_cosmos_client():
    """Initialize and return a Cosmos DB client."""
    client = CosmosClient(COSMOS_ENDPOINT, credential=COSMOS_KEY)
    
    # Get or create database
    database = client.get_database_client(COSMOS_DATABASE)
    
    # Get or create container with partition key
    container_name = COSMOS_CONTAINER
    
    try:
        container = database.get_container_client(container_name)
        container.read()
    except Exception:
        # Container doesn't exist, create it
        container = database.create_container(
            id=container_name,
            partition_key=PartitionKey(path="/id"),
            indexing_policy={
                'indexingMode': 'consistent',
                'includedPaths': [
                    {'path': '/*'}
                ],
                'excludedPaths': [
                    {'path': '/embedding/?'}  # Don't index the embedding vector
                ]
            }
        )
    
    return container


def store_embedding(container, text, embedding, metadata):
    """
    Store text, embedding, and metadata in Cosmos DB.
    Performs deduplication based on text hash.
    """
    if not container:
        return False
    
    try:
        # Check if this text hash already exists
        text_hash = metadata.get("text_hash")
        
        if text_hash:
            # Query to check if hash exists
            query = f"SELECT * FROM c WHERE c.metadata.text_hash = '{text_hash}'"
            items = list(container.query_items(
                query=query,
                enable_cross_partition_query=True
            ))
            
            # Skip if hash exists
            if items:
                return False
        
        # Create a unique ID
        document_id = f"{metadata['filename']}_{metadata['chunk_id']}"
        
        # Create the document
        document = {
            'id': document_id,
            'text': text,
            'embedding': embedding,
            'metadata': metadata
        }
        
        # Store in Cosmos DB
        container.upsert_item(document)
        return True
        
    except Exception as e:
        print(f"Error storing embedding: {str(e)}")
        return False


def query_similar_documents(container, query_embedding, top_k=5):
    """
    Query for similar documents using vector similarity.
    Note: This is a simplified version. Cosmos DB requires special 
    vector search setup or custom scripts for vector similarity.
    """
    # This would typically be implemented using Cosmos DB's 
    # vector search capabilities or a custom script
    # Implementing basic functionality here for reference
    
    # Convert query embedding to string for the query
    embedding_str = json.dumps(query_embedding)
    
    # This is a placeholder - in a real implementation, you would:
    # 1. Use Cosmos DB's vector search if available
    # 2. Or retrieve all documents and compute cosine similarity locally
    # 3. Or use a stored procedure in Cosmos DB
    
    # Placeholder query - this won't work as written
    # You'll need to adapt this based on your Cosmos DB setup
    query = """
    SELECT TOP @top_k c.id, c.text, c.metadata
    FROM c
    """
    
    parameters = [{"name": "@top_k", "value": top_k}]
    
    items = list(container.query_items(
        query=query,
        parameters=parameters,
        enable_cross_partition_query=True
    ))
    
    return items