# 📄 Ask your PDF: a RAG app with hybrid search

**Live demo:** https://bigbrainpart-ask-your-pdf.streamlit.app/

Upload a PDF and ask questions about it. Answers come only from the document, with page citations.

Built step by step to learn Retrieval-Augmented Generation (RAG): LangChain, FAISS, sentence-transformers, BM25, Gemini and Streamlit.

## Features

- **Upload any text PDF** and ask questions in a chat interface
- **Hybrid search**: meaning search (FAISS + all-MiniLM-L6-v2) combined with keyword search (BM25) using Reciprocal Rank Fusion
- **Grounded answers** from Gemini, citing page numbers, and saying "I couldn't find that" when the PDF doesn't cover it
- **Sources panel** showing which chunks were used and how each was found (meaning, keywords, or both)
- **Streaming answers**, cached indexes per file (SHA-256 of the file + embedding model), and an off-topic cutoff that skips the LLM call
- **Safety checks**: image-only PDFs get a clear error; reference sections are removed before indexing

## Evaluation

`eval.py` measures retrieval: does the right page reach the top 4 chunks sent to the LLM?

| Test set | Meaning only | Hybrid |
|---|---|---|
| Original (15 questions) | 14/15 (93%) | **15/15 (100%)** |
| Held-out, written after the fix (7 questions) | 6/7 (86%) | **7/7 (100%)** |

Misses with meaning-only search were short questions about rare names (e.g. "What is MAESTRO?"); keyword search fixes them.
The off-topic cutoff (`MAX_DISTANCE = 1.65`) was chosen from these measurements.

**Limits:** small test sets; a "hit" means the right page, not necessarily the right chunk.

## How it works

1. **Load and split**: the PDF is read with PyPDF, the reference section is removed, and the text is split into ~1000-character chunks
2. **Index**: chunks are embedded with all-MiniLM-L6-v2 into FAISS and indexed with BM25
3. **Retrieve**: both searches run, and their rankings are fused with RRF
4. **Generate**: the top 4 chunks and the question go to Gemini with a prompt that allows only answers from the context

## Run locally

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file with your free key from [Google AI Studio](https://aistudio.google.com):

```text
GOOGLE_API_KEY=your-key-here
```

Then:

```bash
streamlit run app.py
```

## Project files

| File | Purpose |
|---|---|
| `app.py` | Streamlit web app |
| `rag_core.py` | Shared logic: loading, splitting, hybrid search, prompt |
| `eval.py` | Retrieval evaluation |
| `load.py`, `embed.py`, `search.py`, `ask.py` | Step-by-step learning scripts |

## Credits

`sample.pdf` is the Wikipedia article "AI agent", used under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).