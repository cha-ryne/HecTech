import sys
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import openai

# 1. Read PDF and extract text
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

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
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:8]  # Retrieve more chunks for richer context
    return [chunks[i] for i in top_indices]

# 5. Generate answer using Llama 3
def generate_answer(context, query, generator):
    # Truncate context to 800 characters for prompt
    truncated_context = context[:1200]  # Slightly longer context
    prompt = (
        f"Context: {truncated_context}\n\n"
        "Based only on the above context, what were the main findings or results of the dissertation? "
        "Please cite specific statistics, relationships, and p-values if available. Do not make assumptions beyond the provided text."
    )
    client = openai.OpenAI(
        base_url="http://localhost:1234/v1",
        api_key="lm-studio"
    )
    chat_response = client.chat.completions.create(
        model="phi-3-mini-4k-instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Only answer from the provided context. Cite statistics and p-values if available. Do not make assumptions beyond the text."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=800,
        temperature=0.7
    )
    return chat_response.choices[0].message.content.strip()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python rag_llama3.py <pdf_path> <question>")
        sys.exit(1)
    pdf_path = sys.argv[1]
    question = sys.argv[2]

    print("Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)
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

    generator = None  # Not used, kept for compatibility

    print("Generating answer...")
    answer = generate_answer(context, question, generator)
    print("\n---\nAnswer:\n", answer)
