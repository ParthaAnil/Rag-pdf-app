import hashlib
import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

from rag_core import (
    EMBEDDING_MODEL, LLM_MODEL, PROMPT, build_bm25, format_context, hybrid_search, load_and_split,
)

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

TOP_K = 4
MAX_DISTANCE = 1.65  # from eval.py: real questions <= 1.61, most off-topic >= 1.71
NOT_FOUND = "I couldn't find that in the document."

st.set_page_config(page_title="Ask your PDF", page_icon="📄")
st.title("📄 Ask your PDF")
if not os.getenv("GOOGLE_API_KEY"):
    st.error("No GOOGLE_API_KEY found. Add it to .env (local) or to the app's Secrets (Streamlit Cloud).")
    st.stop()


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0, timeout=60, max_retries=2)


@st.cache_resource(show_spinner=False)
def build_index(file_hash, embedding_model, _pdf_bytes):
    # PyPDFLoader needs a real file, so write the upload to a temporary folder
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        pdf_path = Path(tmp) / "upload.pdf"
        pdf_path.write_bytes(_pdf_bytes)
        chunks = load_and_split(pdf_path)
        return FAISS.from_documents(chunks, get_embeddings()), chunks, build_bm25(chunks)


def show_sources(sources):
    with st.expander(f"Sources checked ({len(sources)})"):
        for source in sources:
            st.markdown(f"**Page {source['page']}** · found by {source.get('found_by', 'meaning')}")
            st.text(source["text"])


# --- 1. Upload ---
uploaded = st.file_uploader("Upload a PDF", type="pdf")
if uploaded is None:
    st.info("Upload a PDF to start asking questions.")
    st.stop()

pdf_bytes = uploaded.getvalue()
file_hash = hashlib.sha256(pdf_bytes).hexdigest()

# --- 2. Index (cached: the same file + model is only embedded once) ---
try:
    with st.spinner("Reading and indexing the PDF..."):
        vectorstore, chunks, bm25 = build_index(file_hash, EMBEDDING_MODEL, pdf_bytes)
except ValueError as e:
    st.error(str(e))
    st.stop()

st.caption(f"Indexed **{uploaded.name}** into {len(chunks)} chunks.")

# A new PDF starts a new chat
if st.session_state.get("file_hash") != file_hash:
    st.session_state.file_hash = file_hash
    st.session_state.messages = []

# --- 3. Show the chat so far ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            show_sources(message["sources"])

# --- 4. Handle a new question ---
question = st.chat_input("Ask a question about the PDF")
if question and question.strip():
    question = question.strip()
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
                results, best_distance = hybrid_search(question, vectorstore, bm25, chunks, k=TOP_K)
    sources = [
            {
                "page": doc.metadata.get("page_label", "?"),
                "text": doc.page_content,
                "found_by": found_by,
            }
            for doc, found_by in results
        ]

    if best_distance > MAX_DISTANCE:
        # Nothing is close enough: answer instantly, without calling Gemini
        answer = NOT_FOUND
        st.markdown(answer)
    else:
        docs = [doc for doc, _ in results]
        chain = PROMPT | get_llm()
        try:
            stream = chain.stream({"context": format_context(docs), "question": question})
            answer = st.write_stream(chunk.text for chunk in stream)
        except Exception as e:
                answer = f"Gemini error: {e}"
                st.error(answer)

        show_sources(sources)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})