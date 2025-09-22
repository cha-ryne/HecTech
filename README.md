# RAG with Llama 3 Demo

This project demonstrates Retrieval Augmented Generation (RAG) using Llama 3 and a PDF document.

## Features
- Reads and extracts text from a PDF
- Chunks and embeds the document
- Retrieves relevant chunks based on a user query
- Generates an answer using Llama 3, augmented by retrieved context

## Requirements
- Python 3.8+
- See `requirements.txt` for dependencies

## Usage
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Run the script:
   ```
   python rag_llama3.py <path_to_pdf> "Your question here"
   ```

## Notes
- The Llama 3 model (`meta-llama/Meta-Llama-3-8B-Instruct`) is loaded from Hugging Face. You may need to accept their terms and have sufficient hardware.
- For large PDFs, processing may take time.

---
