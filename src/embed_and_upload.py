"""Script to embed documents and upload them to Pinecone vector database.

This module handles the process of converting text documents into embeddings using LlamaCpp
and uploading them to a Pinecone vector database for efficient retrieval.
"""

import os
import json
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.embeddings import LlamaCppEmbeddings

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY", "")
if api_key:
    print(f"API Key loaded (first 4 chars): {api_key[:4]}...")
else:
    print("Warning: PINECONE_API_KEY not found in environment variables")

JSONL_FILE = "langchain_documents.jsonl"
PINECONE_INDEX_NAME = "developer-quickstart-py"
EMBEDDING_DIMENSION = 4096

def clean_metadata(metadata):
    """Convert None values in metadata to empty strings.
    
    Args:
        metadata (dict): Dictionary containing metadata key-value pairs
        
    Returns:
        dict: Cleaned metadata with None values replaced by empty strings
    """
    return {k: "" if v is None else v for k, v in metadata.items()}

def init_pinecone():
    """Initialize and return Pinecone index with specified configuration.
    
    Returns:
        Index: Configured Pinecone index ready for vector operations
    """
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    if pc.has_index(PINECONE_INDEX_NAME):
        pc.delete_index(PINECONE_INDEX_NAME)
        print(f"Deleted existing index: {PINECONE_INDEX_NAME}")
    
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
    print(f"Created new index: {PINECONE_INDEX_NAME}")
    
    return pc.Index(PINECONE_INDEX_NAME)

def load_documents():
    """Load documents from JSONL file and clean their metadata.
    
    Returns:
        list[Document]: List of Document objects with cleaned metadata
    """
    documents = []
    with open(JSONL_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            doc = Document(
                page_content=data['page_content'],
                metadata=clean_metadata(data['metadata'])
            )
            documents.append(doc)
    return documents

def main():
    """Main function to embed documents and upload them to Pinecone.
    
    Initializes LlamaCpp embeddings and Pinecone, then processes documents in batches
    to generate embeddings and upload them to the vector database.
    """
    embeddings = LlamaCppEmbeddings(
        model_path="./models/llama-2-7b-chat.Q4_K_M.gguf",
        n_ctx=2048,
        n_batch=100
    )
    
    index = init_pinecone()
    
    documents = load_documents()
    print(f"Loaded {len(documents)} documents")
    
    batch_size = 100
    for i in tqdm(range(0, len(documents), batch_size)):
        batch = documents[i:i + batch_size]
        
        texts = [doc.page_content for doc in batch]
        embeds = embeddings.embed_documents(texts)
        
        vectors = []
        for j, (doc, embedding) in enumerate(zip(batch, embeds)):
            vectors.append({
                'id': f'doc_{i+j}',
                'values': embedding,
                'metadata': {
                    **clean_metadata(doc.metadata),
                    'text': doc.page_content
                }
            })
        
        index.upsert(vectors=vectors)
    
    print("Successfully embedded and uploaded all documents to Pinecone!")

if __name__ == "__main__":
    main()