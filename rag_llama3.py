import sys
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import openai
import requests
import os

# 1. Read PDF and extract text
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    # Flexible header search: extract between CHAPTER IV and CHAPTER V
    import re
    start_match = re.search(r'chapter\s*iv\s*[:\-]?\s*results?\s*and\s*discussion', text, re.IGNORECASE)
    end_match = re.search(r'chapter\s*v\s*[:\-]?\s*conclusion\s*and\s*recommendations', text, re.IGNORECASE)
    if start_match:
        start_idx = start_match.start()
        end_idx = end_match.start() if end_match else None
        if end_idx:
            filtered_text = text[start_idx:end_idx]
        else:
            filtered_text = text[start_idx:]
        return filtered_text
    else:
        print("\n--- ERROR: 'CHAPTER IV: RESULTS AND DISCUSSION' section not found in PDF. Stopping execution. ---\n")
        return None

# 2. Chunk text
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# 3. Embed chunks
def embed_chunks(chunks, embedder):
    return embedder.encode(chunks)

# 4. Retrieve relevant chunks
def retrieve(query, chunks, chunk_embeddings, embedder, top_k=3):
    query_emb = embedder.encode([query])[0]
    scores = [torch.cosine_similarity(torch.tensor(query_emb), torch.tensor(chunk_emb), dim=0).item() for chunk_emb in chunk_embeddings]
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
    return [chunks[i] for i in top_indices]

# 5. Generate answer using Llama 3
def generate_answer(context, query, generator):
    truncated_context = context[:1500]
    prompt = f"Context: {truncated_context}\n\nQuestion: {query}\nAnswer: "
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": api_key
    }
    data = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 1500
        }
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    result = response.json()
    # Gemini returns answer in result['candidates'][0]['content']['parts'][0]['text']
    return result['candidates'][0]['content']['parts'][0]['text'].strip()

# Prompt chaining for multi-step reasoning
def prompt_chain(initial_prompt, follow_up_prompts, context=None):
    if context is None:
        context = ""
    # Step 1: Run initial prompt
    answer = generate_answer(context, initial_prompt, None)
    context = f"{context}\n{answer}"
    # Step 2: Run each follow-up prompt, updating context
    for prompt in follow_up_prompts:
        answer = generate_answer(context, prompt, None)
        context = f"{context}\n{answer}"
    return answer

if __name__ == "__main__":
    # Example usage for prompt chaining
    initial_prompt = "Summarize the key trends in global temperature changes over the past century."
    follow_up_prompts = [
        "Based on the trends identified, list the major scientific studies that discuss the causes of these changes.",
        "Summarize the findings of the listed studies, focusing on the impact of climate change on marine ecosystems.",
        "Propose three strategies to mitigate the impact of climate change on marine ecosystems based on the summarized findings."
    ]

    # If PDF and question are provided, use RAG context
    if len(sys.argv) >= 3:
        pdf_path = sys.argv[1]
        question = sys.argv[2]
        print("Extracting text from PDF...")
        text = extract_text_from_pdf(pdf_path)
        if text is None:
            print("No relevant section found. Exiting without calling LLM.")
            sys.exit(1)
        print("Chunking text...")
        chunks = chunk_text(text)
        print(f"Total chunks: {len(chunks)}")
        print("Loading embedding model...")
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        print("Embedding chunks...")
        chunk_embeddings = embed_chunks(chunks, embedder)
        print("Retrieving relevant chunks...")
        relevant_chunks = retrieve(question, chunks, chunk_embeddings, embedder)
        context = "\n".join(relevant_chunks)
        generator = None
        print("Generating answer...")
        answer = generate_answer(context, question, generator)
        print("\n---\nAnswer:\n", answer)
    else:
        # Demo: run prompt chain
        final_result = prompt_chain(initial_prompt, follow_up_prompts)
        print("Final result:", final_result)
