from dotenv import load_dotenv
import os
from pathlib import Path
import numpy as np
import google.genai as genai
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from simple_vector_store import SimpleVectorStore
from sklearn.metrics.pairwise import cosine_similarity
from read_info_pdf import ingest_uploaded_files
import time
load_dotenv()

DATA_DIR = Path("data")
INDEX_PATH = Path("vector_store.pk")
INDEX_PATH1 = Path("vector_store1.pk1")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

@st.cache_resource
def load_embedding_model():
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

def generate_answer_with_gemini(query, contexts):

    
    # key = gemini_api_key
    # if not key:
    #     return None
    prompt = (
        "Use the provided context to answer the question concisely. "
        "If the answer cannot be found in the context, say you could not find a confident answer.\n\n"
        f"Context:\n{contexts}\n\nQuestion: {query}\nAnswer:"
    )
    try:
        client = genai.Client()
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.4,
                max_output_tokens=80,
            ),
        )
        return response.text.strip()
    except Exception as exc:
        st.error(f"Gemini request failed: {exc}")
        return None

def main():
    if "file" not in st.session_state:
        st.session_state.file = False

    if "chunk_count" not in st.session_state:
        st.session_state.chunk_count = 0

    if not st.session_state.file:
        with st.spinner("Loading..."):
            time.sleep(3)
    st.set_page_config(page_title="RAG System", layout="wide")
    st.title( "RAG System")
    st.write(
        "Upload documents, build a local semantic index, and ask questions with retrieval-augmented responses."
    )

    model = load_embedding_model()
    store = SimpleVectorStore.load((INDEX_PATH))
    store1 = SimpleVectorStore.load((INDEX_PATH1))
    max_countchunks=0
    

    if not st.session_state.file:
        # print(st.session_state.file)
        st.header("Index Management")
        st.write("Upload `.txt` or `.pdf` files and persist the semantic index locally.")
        uploaded_files = st.file_uploader("Upload documents", type=["txt", "pdf"], accept_multiple_files=True)
        ingest_button = st.button("Ingest uploaded files")
        if ingest_button and uploaded_files:
            with st.spinner("Embedding documents..."):
                count = ingest_uploaded_files(uploaded_files, store, model)
                store.save(INDEX_PATH)
                count1 = ingest_uploaded_files(uploaded_files, store1, model)
                store1.save(INDEX_PATH1)
                st.session_state.chunk_count = min(count, count1)

            st.success(f"Ingested {st.session_state.chunk_count} text chunks into vector store.")
            st.session_state.file=True
            st.rerun()
        elif ingest_button and not uploaded_files:
            st.warning("Select files to upload before ingesting.")
        if INDEX_PATH.exists():
            if st.button("Clear stored index"):
                INDEX_PATH.unlink(missing_ok=True)
                store = SimpleVectorStore()
                st.success("Local vector store cleared.")

        st.markdown("---")
        st.header("Gemini Settings")
        gemini_api_key = st.text_input("Gemini API Key", type="password", placeholder="AIza...", help="Optional. Needed for answer generation.")
        st.caption("If not provided, only retrieved document chunks will be shown.")

    else:
        st.header("Ask a question")
        query = st.text_input("Enter your question")
        top_k = st.slider("Number of retrieved chunks", min_value=1, max_value=max(st.session_state.chunk_count,6), value=4)

        if st.button("Search"):
            if store.embeddings.size == 0:
                st.warning("No vector store available. Ingest documents first.")
                return
            if not query:
                st.warning("Enter a question before searching.")
                return

            with st.spinner("Retrieving relevant context..."):
                start_time = time.time()
                results = store.search(query, model, top_k=top_k)
                contexts = "\n\n".join([f"Source: {item['source']}\n{item['text']}" for item in results])
                search_time = time.time() - start_time
                st.write(f"All the context creation and the searching is completed in {search_time:.3f} seconds for normal chunking methord model")
                
                start_time = time.time()
                results = store1.search(query, model, top_k=top_k)
                contexts1 = "\n\n".join([f"Source: {item['source']}\n{item['text']}" for item in results])
                search_time = time.time() - start_time
                st.write(f"All the context creation and the searching is completed in {search_time:.3f} seconds for sementic chunking bassed model")


           
            st.subheader("Answer")
            answer = generate_answer_with_gemini(query, contexts)
            answer1 = generate_answer_with_gemini(query, contexts1)
            if answer and answer1:
                st.markdown(
                    "<h4 style='color:red;'>With normal chunking we are getting the following answer</h4>",
                    unsafe_allow_html=True
                )
                st.write(answer)

                st.markdown(
                    "<h4 style='color:red;'>With semantic chunking we are getting the following answer</h4>",
                    unsafe_allow_html=True
                )
                st.write(answer1)


            else:
                st.info("Gemini API key is not configured or answer generation failed. Showing retrieved context below.")

    
            st.subheader("Retrieved chunks")
            for idx, item in enumerate(results, 1):
                st.markdown(f"**{idx}. Source:** {item['source']}  \n**Score:** {item['score']:.3f}")
                st.write(item['text'])

        st.markdown("---")
        st.subheader("Usage notes")
        st.markdown(
            "- Upload text or PDF documents in the sidebar.\n"
            "- Ingest creates a local semantic index in `vector_store.pkl`.\n"
            "- Ask a question and retrieve the most relevant chunks.\n"
            "- If you provide a Gemini API key, Streamlit will use it to generate an answer from the retrieved contexts."
        )


if __name__ == "__main__":
    main()
