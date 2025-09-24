import os
import glob
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests

# 1. Extract and chunk text from all PDFs in a folder
def extract_and_chunk_pdfs(pdf_folder, chunk_size=500):
    import pytesseract
    from pdf2image import convert_from_path
    pdf_files = glob.glob(os.path.join(pdf_folder, '*.pdf'))
    all_chunks = []
    metadata = []
    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path}")
        reader = PdfReader(pdf_path)
        text = ""
        py_pdf2_pages = 0
        ocr_pages = 0
        # Extract metadata
        doc_info = reader.metadata if hasattr(reader, 'metadata') else reader.documentInfo
        title = getattr(doc_info, 'title', None) or doc_info.get('/Title', '') if doc_info else ''
        author = getattr(doc_info, 'author', None) or doc_info.get('/Author', '') if doc_info else ''
        # Try to extract publication year from metadata
        publication_year = ''
        if doc_info:
            # Try /CreationDate or /ModDate
            for key in ['/CreationDate', '/ModDate']:
                date_val = doc_info.get(key, '') if isinstance(doc_info, dict) else getattr(doc_info, key, '')
                # Convert to string if not already
                if date_val:
                    date_str = str(date_val)
                    import re
                    match = re.search(r'(\d{4})', date_str)
                    if match:
                        publication_year = match.group(1)
                        break
        # If missing, try to parse year from first page
        first_page_text = ""
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
                py_pdf2_pages += 1
                if page_num == 0:
                    first_page_text = page_text
        if not title and first_page_text:
            # Try to guess title from first line
            title = first_page_text.split('\n')[0][:120]
        if not author and first_page_text:
            # Try to guess author from lines containing 'author' or similar
            for line in first_page_text.split('\n'):
                if 'author' in line.lower():
                    author = line.strip()[:120]
                    break
        if not publication_year and first_page_text:
            import re
            match = re.search(r'(19\d{2}|20\d{2})', first_page_text)
            if match:
                publication_year = match.group(1)
        # If no text extracted, try OCR
        if not text.strip():
            print(f"  No text found with PyPDF2, trying OCR...")
            try:
                images = convert_from_path(pdf_path)
                for img_num, img in enumerate(images):
                    ocr_text = pytesseract.image_to_string(img)
                    if ocr_text.strip():
                        text += ocr_text + "\n"
                        ocr_pages += 1
            except Exception as e:
                print(f"  OCR failed: {e}")
        print(f"  PyPDF2 extracted text from {py_pdf2_pages} pages.")
        print(f"  OCR extracted text from {ocr_pages} pages.")
        print(f"  Total extracted text length: {len(text)} characters.")
        words = text.split()
        chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
        print(f"  Total chunks created: {len(chunks)}")
        for idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            metadata.append({
                'pdf': os.path.basename(pdf_path),
                'chunk_idx': idx,
                'title': title,
                'author': author,
                'publication_year': publication_year
            })
    return all_chunks, metadata

# 2. Embed all chunks
def embed_chunks(chunks, embedder):
    return embedder.encode(chunks)

# 3. Build FAISS index
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# 4. Retrieve top-N relevant chunks
def retrieve_top_chunks(query, embedder, index, all_chunks, metadata, top_n=10):
    query_emb = embedder.encode([query])
    D, I = index.search(np.array(query_emb), top_n)
    results = []
    for idx in I[0]:
        results.append({'chunk': all_chunks[idx], 'meta': metadata[idx]})
    return results

# 5. Use Gemini to generate overview from top chunks
def gemini_overview(top_chunks, question, api_key):
    # Add document metadata summary for top 10 relevant documents
    doc_infos = []
    seen_pdfs = set()
    for c in top_chunks:
        meta = c['meta']
        pdf_id = meta['pdf']
        if pdf_id not in seen_pdfs:
            doc_infos.append(f"- Title: {meta.get('title','') or '[Unknown]'}\n  Author: {meta.get('author','') or '[Unknown]'}\n  Year: {meta.get('publication_year','') or '[Unknown]'}\n  File: {pdf_id}")
            seen_pdfs.add(pdf_id)
        if len(doc_infos) >= 10:
            break
    doc_info_str = "Top 10 relevant documents found:\n" + "\n".join(doc_infos) + "\n\n"
    context = "\n\n".join([f"From {c['meta']['pdf']} (chunk {c['meta']['chunk_idx']}): {c['chunk']}" for c in top_chunks])
    prompt = f"{doc_info_str}Context: {context}\n\nQuestion: {question}\nAnswer: "
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
    return result['candidates'][0]['content']['parts'][0]['text'].strip()

if __name__ == "__main__":
    pdf_folder = "theses"  # Change to your folder name
    question = input("Enter your research question: ")
    api_key = os.getenv("GEMINI_API_KEY")
    print("Extracting and chunking PDFs...")
    all_chunks, metadata = extract_and_chunk_pdfs(pdf_folder)
    print(f"Total chunks: {len(all_chunks)}")
    if len(all_chunks) == 0:
        print("No text extracted from any PDFs. Exiting without calling LLM.")
        exit(1)
    print("Loading embedding model...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    print("Embedding chunks...")
    chunk_embeddings = embed_chunks(all_chunks, embedder)
    print("Building FAISS index...")
    index = build_faiss_index(np.array(chunk_embeddings))
    print("Retrieving top relevant chunks...")
    top_chunks = retrieve_top_chunks(question, embedder, index, all_chunks, metadata, top_n=10)
    # Print top 10 relevant documents metadata
    doc_infos = []
    seen_pdfs = set()
    for c in top_chunks:
        meta = c['meta']
        pdf_id = meta['pdf']
        if pdf_id not in seen_pdfs:
            doc_infos.append(f"- Title: {meta.get('title','') or '[Unknown]'}\n  Author: {meta.get('author','') or '[Unknown]'}\n  Year: {meta.get('publication_year','') or '[Unknown]'}\n  File: {pdf_id}")
            seen_pdfs.add(pdf_id)
        if len(doc_infos) >= 10:
            break
    print("\nTop 10 relevant documents found:")
    print("\n".join(doc_infos))
    print("\nGenerating overview with Gemini...")
    overview = gemini_overview(top_chunks, question, api_key)
    print("\n---\nOverview:\n", overview)
