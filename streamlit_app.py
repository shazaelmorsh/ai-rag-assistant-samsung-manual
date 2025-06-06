"""Streamlit web application for Samsung phone support using RAG technology.

This application provides a user interface for querying Samsung product information
using a combination of LlamaCpp embeddings, Pinecone vector store, and OpenAI for
text generation.
"""

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
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("developer-quickstart-py")

embeddings = LlamaCppEmbeddings(
    model_path="./models/llama-2-7b-chat.Q4_K_M.gguf",
    n_ctx=2048,
    n_batch=100
)

llm = OpenAI(temperature=0)

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    text_key="text"
)

def get_unique_metadata_values(field: str) -> List[str]:
    """Retrieve unique values for a specific metadata field from Pinecone vectors.
    
    Args:
        field (str): Name of the metadata field to get unique values for
        
    Returns:
        List[str]: Sorted list of unique non-empty values for the specified field
    """
    vector_sample = index.query(
        vector=[0] * 4096,
        top_k=100,
        include_metadata=True
    )
    
    all_values: Set[str] = set()
    for match in vector_sample['matches']:
        if field in match['metadata']:
            value = match['metadata'][field]
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

metadata_filter: Dict = {}
if selected_models:
    metadata_filter["model_names"] = {"$in": selected_models}
if selected_android != "All":
    metadata_filter["android_version"] = selected_android
if selected_oneui != "All":
    metadata_filter["oneui_version"] = selected_oneui
if selected_features:
    metadata_filter["features"] = {"$in": selected_features}

retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 5,
        "filter": metadata_filter if metadata_filter else None
    }
)

prompt = PromptTemplate.from_template("""You are a Samsung phone support expert. Answer the following question based on the provided context.
If you cannot find a specific answer in the context, say so - do not make up information.

Context: {context}
Question: {input}

Answer the question in a clear and concise way. If relevant, mention specific Samsung features, settings, or menu locations.
""")

document_chain = create_stuff_documents_chain(llm, prompt)
qa_chain = create_retrieval_chain(retriever, document_chain)

user_question = st.text_input("Your question:")

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