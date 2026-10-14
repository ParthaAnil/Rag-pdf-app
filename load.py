from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = "sample.pdf"

loader = PyPDFLoader(PDF_PATH)
pages = loader.load()
print(f"Loaded {len(pages)} pages")

total_chars = sum(len(page.page_content) for page in pages)
print(f"Total characters: {total_chars}")
if total_chars == 0:
    raise SystemExit("No text found. This PDF is only images. Re-save it with real text.")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(pages)
print(f"Split into {len(chunks)} chunks")

print("\n--- First chunk ---")
print(chunks[0].page_content)
print("\n--- Its metadata ---")
print(chunks[0].metadata)