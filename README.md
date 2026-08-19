# 🎓 Student Management RAG Assistant

A Retrieval-Augmented Generation app that answers questions about
student documents (handbooks, notices, circulars, policies). Embeddings
run locally and free (`sentence-transformers`); answer generation uses
**Llama 3 hosted on Groq**, so the app works for anyone visiting the
public link — no local model server required.

## How it works
1. You upload PDFs in the sidebar.
2. Text is split into chunks and embedded locally (`sentence-transformers`).
3. Chunks are stored in a lightweight NumPy vector index.
4. When you ask a question, the most relevant chunks are retrieved.
5. Llama 3 (via the Groq API) reads those chunks and writes a grounded
   answer — it won't make things up outside the documents.

## Setup

### 1. Get a free Groq API key
Sign up at https://console.groq.com and create an API key.

- **Running locally:** set it as an environment variable:
  ```bash
  export GROQ_API_KEY="your-key-here"     # Windows: set GROQ_API_KEY=your-key-here
  ```
- **Deployed on Streamlit Community Cloud:** go to your app ->
  **Manage app -> Settings -> Secrets** and add:
  ```toml
  GROQ_API_KEY = "your-key-here"
  ```

### 2. Install Python dependencies
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

### 4. Use it
- In the sidebar, upload one or more PDFs (student handbook, exam
  notice, fee circular, etc.)
- Click **Build / Rebuild Index**
- Ask questions in the chat box, e.g.:
  - "What is the last date to pay semester fees?"
  - "What is the attendance requirement to sit for exams?"
  - "How do I apply for a hostel room?"

The index is saved to `faiss_index/` on disk, so next time you can
just click **Load Saved Index** instead of re-uploading everything.

## Customizing the answers
Open `rag_core.py` and edit `SYSTEM_PROMPT` — this controls the
assistant's persona, tone, and rules (e.g. make it stricter, add a
language preference, make it always mention a contact office, etc.).

You can also:
- Swap `OLLAMA_MODEL` in `rag_core.py` to another Groq-hosted model
  (e.g. `llama-3.3-70b-versatile` for higher quality, or keep
  `llama-3.1-8b-instant` for speed)
- Change `CHUNK_SIZE` / `CHUNK_OVERLAP` in `rag_core.py` for how documents are split
- Change `k` (chunks retrieved) in the Streamlit sidebar slider

## Project structure
```
student-rag/
├── app.py          # Streamlit UI (upload, build index, chat)
├── rag_core.py      # RAG pipeline: load, chunk, embed, retrieve, generate
├── requirements.txt
└── README.md
```

## Extending it further
- Add CSV/Excel student records with `CSVLoader` / `UnstructuredExcelLoader`
- Add user authentication so students only see their own record-based answers
- Add a "clear index" button and multi-session support
- Swap FAISS for Chroma or a hosted vector DB if you need persistence across
  many users
