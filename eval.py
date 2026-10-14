from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from rag_core import EMBEDDING_MODEL, build_bm25, hybrid_search, load_and_split

BASE_DIR = Path(__file__).parent
PDF_PATH = BASE_DIR / "sample.pdf"
TOP_K = 4

## Questions written AFTER the hybrid fix, to check for overfitting
NEW_CASES = [
    ("who is Milind tambe?", {"1"}),
    ("what Thomas shedd did in February 2025?", {"4"}),
    ("what google did to operating systems?", {"4"}),
    ("what arguments were done by CNN?", {"6"}),
    ("what are the main threats that can be done by ai to humans?", {"5"}),
    ("Do you think AI will replace humans?", {"5", "6"}),
    ("What is software agents?", {"1", "8"}),
    ("Can I control an automated AI?", set()),
    ("Thoughts about AI Automation?", set()),
    ("Do you Indian prime minister?", set()),
]

# Build the pipeline fresh, so the eval always tests the current code
chunks = load_and_split(PDF_PATH)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = FAISS.from_documents(chunks, embeddings)
bm25 = build_bm25(chunks)


def pages_of(docs):
    return {doc.metadata.get("page_label", "?") for doc in docs}


semantic_hits = 0
hybrid_hits = 0
answerable_best = []
off_topic_best = []

print(f"{'best':>5}  {'meaning':<8} {'hybrid':<8} question")
for question, expected_pages in NEW_CASES:
    semantic_docs = [doc for doc, _ in vectorstore.similarity_search_with_score(question, k=TOP_K)]
    hybrid_results, best = hybrid_search(question, vectorstore, bm25, chunks, k=TOP_K)
    hybrid_docs = [doc for doc, _ in hybrid_results]

    if expected_pages:
        answerable_best.append(best)
        semantic_hit = bool(expected_pages & pages_of(semantic_docs))
        hybrid_hit = bool(expected_pages & pages_of(hybrid_docs))
        semantic_hits += semantic_hit
        hybrid_hits += hybrid_hit
        semantic_result = "HIT" if semantic_hit else "MISS"
        hybrid_result = "HIT" if hybrid_hit else "MISS"
    else:
        off_topic_best.append(best)
        semantic_result = hybrid_result = "OFF"
    print(f"{best:5.2f}  {semantic_result:<8} {hybrid_result:<8} {question}")

n = len(answerable_best)
print(f"\nHit rate, meaning only: {semantic_hits}/{n}")
print(f"Hit rate, hybrid:       {hybrid_hits}/{n}")
print(f"Answerable best distances: {min(answerable_best):.2f} to {max(answerable_best):.2f}")
print(f"Off-topic best distances:  {min(off_topic_best):.2f} to {max(off_topic_best):.2f}")