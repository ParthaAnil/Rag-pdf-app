from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from rag_core import EMBEDDING_MODEL, load_and_split

BASE_DIR = Path(__file__).parent
PDF_PATH = BASE_DIR / "sample.pdf"
INDEX_DIR = BASE_DIR / "faiss_index"

try:
    chunks = load_and_split(PDF_PATH)
except ValueError as e:
    raise SystemExit(str(e))
print(f"{len(chunks)} chunks ready")

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = FAISS.from_documents(chunks, embeddings)
print(f"FAISS index holds {vectorstore.index.ntotal} vectors")

vectorstore.save_local(str(INDEX_DIR))
print(f"Saved index to {INDEX_DIR}")