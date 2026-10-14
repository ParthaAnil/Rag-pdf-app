from pathlib import Path

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

from rag_core import EMBEDDING_MODEL, LLM_MODEL, PROMPT, format_context

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")
INDEX_DIR = BASE_DIR / "faiss_index"

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)

llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0, timeout=60, max_retries=2)
chain = PROMPT | llm

while True:
    question = input("\nAsk a question (or press Enter to quit): ").strip()
    if not question:
        break

    docs = vectorstore.similarity_search(question, k=4)

    try:
        response = chain.invoke({"context": format_context(docs), "question": question})
    except Exception as e:
        print(f"\nGemini error: {e}")
        continue

    print("\nAnswer:", response.text)
    docs_in_order = sorted(docs, key=lambda doc: doc.metadata.get("page", 0))
    pages = dict.fromkeys(doc.metadata.get("page_label", "?") for doc in docs_in_order)
    print("Sources checked: pages", ", ".join(pages))