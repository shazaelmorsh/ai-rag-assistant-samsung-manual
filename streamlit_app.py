"""Streamlit web application for Samsung phone support using RAG technology.

This application provides a user interface for querying Samsung product information
using a combination of TF-IDF retrieval and OpenAI for text generation.
"""

import os
import json
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from typing import List, Dict, Set
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.retrievers import TFIDFRetriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

load_dotenv()

llm = ChatOpenAI(
    temperature=0,
    model="openai/gpt-4o-mini-2024-07-18",
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://github.com/shazaelmorsh/ai-rag-assistant-samsung-manual",
        "X-Title": "Samsung AI RAG Assistant"
    }
)

# Load documents from JSONL file
def load_documents() -> List[Document]:
    documents = []
    with open("langchain_documents.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            doc = Document(
                page_content=data['page_content'],
                metadata=data['metadata']
            )
            documents.append(doc)
    return documents

# Initialize TF-IDF retriever
documents = load_documents()
retriever = TFIDFRetriever.from_documents(
    documents,
    k=5  # Number of documents to retrieve
)

def get_unique_metadata_values(field: str) -> List[str]:
    """Retrieve unique values for a specific metadata field from documents.
    
    Args:
        field (str): Name of the metadata field to get unique values for
        
    Returns:
        List[str]: Sorted list of unique non-empty values for the specified field
    """
    all_values: Set[str] = set()
    for doc in documents:
        if field in doc.metadata:
            value = doc.metadata[field]
            if isinstance(value, list):
                all_values.update(value)
            else:
                all_values.add(value)
    
    return sorted(list(filter(None, all_values)))

st.title("Samsung Phone Assistant 📱")
st.write("Ask any question about your Samsung phone!")

st.sidebar.title("Filters")

models = get_unique_metadata_values("model_names")
android_versions = get_unique_metadata_values("android_version")
oneui_versions = get_unique_metadata_values("oneui_version")
features = get_unique_metadata_values("features")

selected_models = st.sidebar.multiselect("Select Phone Model(s)", models)
selected_android = st.sidebar.selectbox("Android Version", ["All"] + android_versions)
selected_oneui = st.sidebar.selectbox("One UI Version", ["All"] + oneui_versions)
selected_features = st.sidebar.multiselect("Features", features)

# Filter documents based on metadata
filtered_documents = documents
if selected_models or selected_android != "All" or selected_oneui != "All" or selected_features:
    filtered_documents = []
    for doc in documents:
        if selected_models and not any(model in doc.metadata.get("model_names", []) for model in selected_models):
            continue
        if selected_android != "All" and doc.metadata.get("android_version") != selected_android:
            continue
        if selected_oneui != "All" and doc.metadata.get("oneui_version") != selected_oneui:
            continue
        if selected_features and not any(feature in doc.metadata.get("features", []) for feature in selected_features):
            continue
        filtered_documents.append(doc)
    
    # Update retriever with filtered documents
    retriever = TFIDFRetriever.from_documents(
        filtered_documents,
        k=5
    )

prompt = PromptTemplate.from_template("""You are a Samsung phone support expert. Answer the following question based on the provided context.
If you cannot find a specific answer in the context, say so - do not make up information.

Context: {context}
Question: {input}

Answer the question in a clear and concise way. If relevant, mention specific Samsung features, settings, or menu locations.
""")

document_chain = create_stuff_documents_chain(llm, prompt)
qa_chain = create_retrieval_chain(retriever, document_chain)

user_question = st.text_input("Your question:", max_chars=50)

if user_question:
    with st.spinner("Finding answer..."):
        response = qa_chain.invoke({"input": user_question})
        
        st.write("### Answer")
        st.write(response["answer"])
        
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