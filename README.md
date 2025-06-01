# AI RAG Assistant

## Overview
The AI RAG Assistant is a cutting-edge solution designed to enhance customer support and self-service capabilities for Samsung products. By leveraging Retrieval-Augmented Generation (RAG) technology, this application provides instant, accurate, and contextually relevant answers to customer inquiries using Samsung's official product manuals. 

### Target Audience:
- **End Users**: Samsung customers seeking quick answers to product-related questions.
- **Support Teams**: Customer service representatives who can use the tool to enhance their efficiency.
- **Business Partners**: Resellers and distributors looking to provide better support to their customers.

## Features
- **Text Chunking**: Breaks down large documents into manageable chunks for processing.
- **Embedding and Uploading**: Converts text chunks into embeddings and uploads them to a model for retrieval.
- **Model Integration**: Supports integration with various models, including the Llama 2 model.

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

## Model
The project includes a pre-trained Llama 2 model located in the `model/` directory. You can use this model for retrieval tasks.

## Documentation
For detailed documentation, refer to the Samsung manuals located in the `samsung_manuals/` directory. These manuals are used to answer client questions and provide accurate information.

## Contributing
Contributions are welcome! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) file for details on how to contribute to this project.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
