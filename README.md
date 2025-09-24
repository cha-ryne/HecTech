# RAG with Llama 3 Demo

This project demonstrates Retrieval Augmented Generation (RAG) using Gemini and PDF documents. It includes a script (`multi_thesis_rag.py`) that reads multiple PDFs, chunks and embeds their text, retrieves relevant chunks for a user query, and generates an answer using an LLM.

## Features
- Reads and extracts text from multiple PDFs (with OCR fallback)
- Chunks and embeds documents using Sentence Transformers
- Retrieves relevant chunks using FAISS vector search
- Generates an answer using Gemini (Google Generative Language API) with context from retrieved chunks
- Prints metadata for top relevant documents


