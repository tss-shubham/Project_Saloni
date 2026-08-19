"""
Core RAG (Retrieval-Augmented Generation) logic for the
Student Management Assistant.

Pipeline:
1. Load PDF documents (student handbooks, notices, records, etc.)
2. Split them into overlapping chunks
3. Embed chunks locally with a HuggingFace sentence-transformer
4. Store/search embeddings in a simple NumPy-based vector store
   (no compiled C-extension dependency like FAISS, so it installs
   cleanly on any Python version, including brand-new ones)
5. Retrieve the most relevant chunks for a question
6. Feed question + chunks to a local Llama model (via Ollama) to
   generate a grounded, customized answer
"""

import os
import pickle
from typing import List

import numpy as np
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # local, free, no API key
OLLAMA_MODEL = "llama-3.1-8b-instant"   # a Groq-hosted model (name kept for app.py compatibility)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
INDEX_DIR = "faiss_index"        # kept name for backwards compatibility


def _get_groq_api_key() -> str:
    """Read the Groq API key from Streamlit secrets or an env var."""
    key = st.secrets.get("GROQ_API_KEY", None) if hasattr(st, "secrets") else None
    key = key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to Streamlit secrets "
            "(Manage app -> Settings -> Secrets) or as an environment variable."
        )
    return key

SYSTEM_PROMPT = """You are a helpful Student Management Assistant for a school/college.
Answer the student's or staff's question using ONLY the context below,
which comes from official documents (handbooks, notices, circulars).

Rules:
- If the answer is not in the context, say you don't have that information
  and suggest who they could contact (e.g. the admin office).
- Be concise, friendly, and specific (mention dates, fees, deadlines, rules
  exactly as written when relevant).
- Never make up policies, deadlines, or numbers that are not in the context.

Context:
{context}
"""


def load_and_split_pdfs(pdf_paths: List[str]):
    """Load a list of PDF file paths and split them into chunks."""
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs = loader.load()  # one Document per page, with metadata
        for d in docs:
            d.metadata["source"] = os.path.basename(path)
        chunks = splitter.split_documents(docs)
        all_chunks.extend(chunks)
    return all_chunks


class SimpleVectorStore:
    """
    A minimal in-memory vector store using NumPy cosine similarity.
    Avoids compiled dependencies (like faiss-cpu) that may not have
    wheels for very new Python versions yet.
    """

    def __init__(self, embedder: HuggingFaceEmbeddings):
        self.embedder = embedder
        self.documents: List[Document] = []
        self.vectors: np.ndarray | None = None

    def add_documents(self, docs: List[Document]):
        texts = [d.page_content for d in docs]
        vecs = np.array(self.embedder.embed_documents(texts), dtype=np.float32)
        # normalize for cosine similarity via dot product
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        vecs = vecs / norms

        self.documents.extend(docs)
        self.vectors = vecs if self.vectors is None else np.vstack([self.vectors, vecs])

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        if self.vectors is None or len(self.documents) == 0:
            return []
        q_vec = np.array(self.embedder.embed_query(query), dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm
        scores = self.vectors @ q_vec
        top_k = np.argsort(-scores)[:k]
        return [self.documents[i] for i in top_k]

    def save_local(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "store.pkl"), "wb") as f:
            pickle.dump({"documents": self.documents, "vectors": self.vectors}, f)

    @classmethod
    def load_local(cls, path: str, embedder: HuggingFaceEmbeddings):
        store = cls(embedder)
        with open(os.path.join(path, "store.pkl"), "rb") as f:
            data = pickle.load(f)
        store.documents = data["documents"]
        store.vectors = data["vectors"]
        return store


def build_vectorstore(chunks, save_path: str = INDEX_DIR) -> SimpleVectorStore:
    """Embed chunks and build/persist a vector store."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    store = SimpleVectorStore(embeddings)
    store.add_documents(chunks)
    store.save_local(save_path)
    return store


def load_vectorstore(save_path: str = INDEX_DIR) -> SimpleVectorStore:
    """Load a previously saved vector store from disk."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return SimpleVectorStore.load_local(save_path, embeddings)


def format_docs(docs: List[Document]) -> str:
    """Turn retrieved chunks into a single context string with sources."""
    return "\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')} | "
        f"page {d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
    )


class RagChain:
    """Simple retriever + Llama generation wrapper (no LCEL needed)."""

    def __init__(self, store: SimpleVectorStore, model_name: str = OLLAMA_MODEL, k: int = 4):
        self.store = store
        self.k = k
        self.llm = ChatGroq(
            model=model_name,
            temperature=0.2,
            api_key=_get_groq_api_key(),
        )

    def retrieve(self, question: str) -> List[Document]:
        return self.store.similarity_search(question, k=self.k)

    def invoke(self, question: str) -> str:
        docs = self.retrieve(question)
        context = format_docs(docs)
        system = SYSTEM_PROMPT.format(context=context)
        response = self.llm.invoke([
            ("system", system),
            ("human", question),
        ])
        return response.content


def build_rag_chain(vectorstore: SimpleVectorStore, model_name: str = OLLAMA_MODEL, k: int = 4):
    """Build the retrieval + Llama generation chain."""
    chain = RagChain(vectorstore, model_name=model_name, k=k)
    return chain, chain  # chain also acts as the "retriever" (has .retrieve)


def answer_question(chain: RagChain, retriever: RagChain, question: str):
    """Run the chain and also return the raw source chunks (for citations)."""
    answer = chain.invoke(question)
    sources = retriever.retrieve(question)
    return answer, sources
