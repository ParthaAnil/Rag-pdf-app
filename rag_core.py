import re

from rank_bm25 import BM25Okapi
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gemini-3.1-flash-lite"

# A heading line that marks the end of the real article
END_HEADINGS = re.compile(r"^\s*(See also|References)\s*$", re.MULTILINE)

RRF_K = 60  # standard constant for Reciprocal Rank Fusion

# Filler words that keyword search should ignore
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "what", "who", "which", "how",
    "why", "when", "where", "do", "does", "did", "of", "in", "on", "for", "to", "and",
    "or", "about", "with", "by", "from", "that", "this", "it", "can", "i", "me", "my",
    "you", "your", "tell",
}
PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You answer questions using ONLY the context below, taken from a PDF.\n"
     "If the answer is not in the context, say: \"I couldn't find that in the document.\"\n"
     "Keep answers short and mention the page numbers you used, like (page 3).\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])


def load_and_split(pdf_path):
    """Load a PDF, drop the reference section, and split it into chunks.

    Raises ValueError if the PDF has no usable text.
    """
    pages = PyPDFLoader(str(pdf_path)).load()

    total_chars = sum(len(page.page_content) for page in pages)
    if total_chars == 0:
        raise ValueError("No text found. This PDF is only images. Re-save it with real text.")

    kept_pages = []
    for page in pages:
        match = END_HEADINGS.search(page.page_content)
        if match:
            page.page_content = page.page_content[:match.start()]
            kept_pages.append(page)
            break
        kept_pages.append(page)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(kept_pages)
    if not chunks:
        raise ValueError("No usable text found before the reference section.")
    return chunks


def format_context(docs):
    """Join chunks into one text block, each labeled with its page."""
    return "\n\n".join(
        f"[page {doc.metadata.get('page_label', '?')}]\n{doc.page_content}" for doc in docs
    )

def tokenize(text):
    """Split text into lowercase keywords: 'What is MAESTRO?' -> ['maestro']"""
    return [word for word in re.findall(r"\w+", text.lower()) if word not in STOPWORDS]


def build_bm25(chunks):
    """Build a keyword-search index over the chunks."""
    return BM25Okapi([tokenize(chunk.page_content) for chunk in chunks])


def hybrid_search(question, vectorstore, bm25, chunks, k=4, candidates=10):
    """Combine meaning search (FAISS) and keyword search (BM25) with Reciprocal Rank Fusion.

    Returns (results, best_distance):
      results       = list of (doc, found_by) pairs, best first
      best_distance = FAISS's closest distance, used for the off-topic cutoff
    """
    # 1. Meaning search
    semantic = vectorstore.similarity_search_with_score(question, k=candidates)
    best_distance = float(semantic[0][1])
    semantic_docs = [doc for doc, _ in semantic]

    # 2. Keyword search: score every chunk, keep the best ones that share a keyword
    scores = bm25.get_scores(tokenize(question))
    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    keyword_docs = [chunks[i] for i in ranked[:candidates] if scores[i] > 0]

    # 3. Reciprocal Rank Fusion: add up 1 / (60 + rank) from each list
    fused = {}
    for found_by, docs in (("meaning", semantic_docs), ("keywords", keyword_docs)):
        for rank, doc in enumerate(docs, start=1):
            entry = fused.setdefault(doc.page_content, {"doc": doc, "score": 0.0, "found_by": []})
            entry["score"] += 1 / (RRF_K + rank)
            entry["found_by"].append(found_by)

    best = sorted(fused.values(), key=lambda entry: entry["score"], reverse=True)[:k]
    return [(entry["doc"], " + ".join(entry["found_by"])) for entry in best], best_distance