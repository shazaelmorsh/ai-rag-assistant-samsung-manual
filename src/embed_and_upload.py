import os
import json
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.embeddings import LlamaCppEmbeddings

# Load environment variables from .env file
load_dotenv()

# Debug: Print API key (masked)
api_key = os.getenv("PINECONE_API_KEY", "")
if api_key:
    print(f"API Key loaded (first 4 chars): {api_key[:4]}...")
else:
    print("Warning: PINECONE_API_KEY not found in environment variables")

# Config
JSONL_FILE = "langchain_documents.jsonl"
PINECONE_INDEX_NAME = "developer-quickstart-py"
EMBEDDING_DIMENSION = 4096  # Llama 2's embedding dimension

def clean_metadata(metadata):
    """Clean metadata by converting None values to empty strings."""
    return {k: "" if v is None else v for k, v in metadata.items()}

def init_pinecone():
    # Initialize Pinecone client
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    # Delete existing index if it exists
    if pc.has_index(PINECONE_INDEX_NAME):
        pc.delete_index(PINECONE_INDEX_NAME)
        print(f"Deleted existing index: {PINECONE_INDEX_NAME}")
    
    # Create new index with standard configuration
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
    documents = []
    with open(JSONL_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            doc = Document(
                page_content=data['page_content'],
                metadata=clean_metadata(data['metadata'])  # Clean metadata before creating Document
            )
            documents.append(doc)
    return documents

def main():
    # Initialize Llama embeddings
    embeddings = LlamaCppEmbeddings(
        model_path="./models/llama-2-7b-chat.Q4_K_M.gguf",
        n_ctx=2048,
        n_batch=100
    )
    
    # Initialize Pinecone
    index = init_pinecone()
    
    # Load documents
    documents = load_documents()
    print(f"Loaded {len(documents)} documents")
    
    # Embed and upload in batches
    batch_size = 100
    for i in tqdm(range(0, len(documents), batch_size)):
        batch = documents[i:i + batch_size]
        
        # Get embeddings for the batch
        texts = [doc.page_content for doc in batch]
        embeds = embeddings.embed_documents(texts)
        
        # Prepare vectors for upload
        vectors = []
        for j, (doc, embedding) in enumerate(zip(batch, embeds)):
            vectors.append({
                'id': f'doc_{i+j}',
                'values': embedding,
                'metadata': {
                    **clean_metadata(doc.metadata),  # Clean metadata again before upload
                    'text': doc.page_content  # Using 'text' consistently
                }
            })
        
        # Upload to Pinecone
        index.upsert(vectors=vectors)
    
    print("Successfully embedded and uploaded all documents to Pinecone!")

if __name__ == "__main__":
    main()
