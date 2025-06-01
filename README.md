# Samsung AI RAG Assistant

## Overview
The Samsung AI RAG Assistant Project is designed to automate customer support for Samsung products. By leveraging Retrieval-Augmented Generation (RAG) technology, this application provides instant, accurate, and contextually relevant answers to customer inquiries using Samsung's official product manuals. 

## 🔧 Technologies

- **LangChain** – for managing the RAG pipeline, preprocessing and embeddings 
- **Streamlit** – for building the user interface  
- **Pinecone** – for vector database indexing and retrieval  

## 📁 Repository Contents

- **`text_chunking.py`** – Splits large product manuals into smaller chunks for processing  
- **`embed_and_upload.py`** – Converts text chunks into embeddings and uploads them to Pinecone  


## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/ai-rag-assistant.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Text Chunking**:
   - Run the `text_chunking.py` script to chunk your document:
     ```bash
     python src/text_chunking.py --input_path path/to/your/document --output_path path/to/save/chunk
     ```

2. **Embedding and Uploading**:
   - Run the `embed_and_upload.py` script to generate embedding and upload it to the model:
     ```bash
     python src/embed_and_upload.py --input_path path/to/chunk --model_path path/to/model
     ```
3. **Run Streamlit app**:
    - ```bash
        streamlit run app.py```

add the pinecode and openai API key inside your .env
## Model
The project includes a pre-trained Llama 2 model located in the `model/` directory. You can use this model for retrieval tasks.

## Data Sources
Place Samsung manuals in the samsung_manuals/ directory. These documents serve as the core knowledge base.
You can download them from Samsung’s official support page:
https://www.samsung.com/us/support/answer/ANS10001611/
