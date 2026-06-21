
import streamlit as st
from PyPDF2 import PdfReader
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer

def chunk_text(text, chunk_size=150, overlap=20):
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = " ".join(tokens[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)

def semantic_chunk_text(text,max_tokens=200):
    sentences = sent_tokenize(text)
    chunks=[]
    current_chunk=[]
    current_tokens=0
    for sentence in sentences:
        token_count=len(tokenizer.encode(sentence, add_special_tokens=False))
        if current_tokens + token_count > max_tokens:
            chunks.append("".join(current_chunk))
            current_chunk = [sentence]
            current_tokens = token_count
        else:
            current_chunk.append(sentence)
            current_tokens +=token_count
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def read_pdf(file_bytes):
    reader = PdfReader(file_bytes)
    text = []
    for page in reader.pages:
        text.append(page.extract_text() or "")
    return "\n\n".join(text)

def read_text_file(file_bytes):
    return file_bytes.read().decode("utf-8", errors="ignore")

def build_metadata(source, text):
    return {"source": source, "text": text}

def ingest_uploaded_files(files, store, model):
    imported_count = 0
    for uploaded_file in files:
        name = uploaded_file.name
        if name.lower().endswith(".pdf"):
            text = read_pdf(uploaded_file)
        elif name.lower().endswith(".txt"):
            text = read_text_file(uploaded_file)
        else:
            st.warning(f"Skipping unsupported file type: {name}")
            continue

        chunks = chunk_text(text)
        store.add(chunks, [build_metadata(name, chunk) for chunk in chunks], model)
        imported_count += len(chunks)

    return imported_count

