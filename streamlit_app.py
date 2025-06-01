import os
import streamlit as st
from pinecone import Pinecone, Index
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from typing import List, Dict, Set
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("developer-quickstart-py")

# Initialize Llama embeddings
embeddings = LlamaCppEmbeddings(
    model_path="./models/llama-2-7b-chat.Q4_K_M.gguf",
    n_ctx=2048,
    n_batch=100
)

# Initialize OpenAI for text generation
llm = OpenAI(temperature=0)

# Create Pinecone retriever
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    text_key="text"
)

# Function to get unique metadata values
def get_unique_metadata_values(field: str) -> List[str]:
    # Query a sample of vectors to get metadata values
    vector_sample = index.query(
        vector=[0] * 4096,  # Updated dimension for Llama
        top_k=100,
        include_metadata=True
    )
    
    all_values: Set[str] = set()
    for match in vector_sample['matches']:
        if field in match['metadata']:
            value = match['metadata'][field]
            if isinstance(value, list):
                # For fields that contain lists (like model_names)
                all_values.update(value)
            else:
                all_values.add(value)
    
    return sorted(list(filter(None, all_values)))

# Streamlit UI
st.title("Samsung Phone Assistant 📱")
st.write("Ask any question about your Samsung phone!")

# Sidebar filters
st.sidebar.title("Filters")

# Get unique values for filters
models = get_unique_metadata_values("model_names")
android_versions = get_unique_metadata_values("android_version")
oneui_versions = get_unique_metadata_values("oneui_version")
features = get_unique_metadata_values("features")

# Filter selections
selected_models = st.sidebar.multiselect("Select Phone Model(s)", models)
selected_android = st.sidebar.selectbox("Android Version", ["All"] + android_versions)
selected_oneui = st.sidebar.selectbox("One UI Version", ["All"] + oneui_versions)
selected_features = st.sidebar.multiselect("Features", features)

# Create metadata filter based on selections
metadata_filter: Dict = {}
if selected_models:
    metadata_filter["model_names"] = {"$in": selected_models}
if selected_android != "All":
    metadata_filter["android_version"] = selected_android
if selected_oneui != "All":
    metadata_filter["oneui_version"] = selected_oneui
if selected_features:
    metadata_filter["features"] = {"$in": selected_features}

# Update retriever with metadata filter
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 5,
        "filter": metadata_filter if metadata_filter else None
    }
)

# Create QA chain with enhanced prompt
template = """You are a helpful Samsung phone support assistant. Use the following pieces of context from Samsung phone manuals to answer the question. If you don't know the answer, just say that you don't know. Don't try to make up an answer.

The information comes from the following context:
{context}

Question: {question}

Answer: """

PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={
        "prompt": PROMPT
    }
)

# User input
user_question = st.text_input("Your question:")

if user_question:
    with st.spinner("Finding answer..."):
        # Get the answer
        response = qa_chain.invoke({
            "query": user_question
        })
        
        # Display the answer
        st.write("### Answer")
        st.write(response["result"])
        
        # Show sources with enhanced metadata
        if st.checkbox("Show sources"):
            st.write("### Sources")
            docs = retriever.get_relevant_documents(user_question)
            for i, doc in enumerate(docs, 1):
                with st.expander(f"Source {i}"):
                    st.write("**Metadata:**")
                    st.write(f"- Models: {', '.join(doc.metadata.get('model_names', []))}")
                    st.write(f"- Section: {doc.metadata.get('section', 'N/A')}")
                    st.write(f"- Page: {doc.metadata.get('page_number', 'N/A')}")
                    st.write(f"- Android Version: {doc.metadata.get('android_version', 'N/A')}")
                    st.write(f"- One UI Version: {doc.metadata.get('oneui_version', 'N/A')}")
                    st.write(f"- Features: {', '.join(doc.metadata.get('features', [])) if doc.metadata.get('features') else 'None'}")
                    st.write("\n**Content:**")
                    st.write(doc.page_content)
