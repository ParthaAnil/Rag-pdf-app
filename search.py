from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

INDEX_DIR = "faiss_index"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.load_local(
    INDEX_DIR,
    embeddings,
    allow_dangerous_deserialization=True,  
)
print(f"Loaded index with {vectorstore.index.ntotal} vectors")

while True:
    question = input("\nAsk a question (or press Enter to quit): ")
    if not question:
        break

    
    results = vectorstore.similarity_search_with_score(question, k=3)

    for rank, (doc, score) in enumerate(results, start=1):
        page = doc.metadata.get("page_label", "?")
        print(f"\n--- Result {rank} | page {page} | distance {score:.3f} ---")
        print(doc.page_content[:300])
        