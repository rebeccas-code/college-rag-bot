from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Iterable, List, Tuple

import numpy as np
from openai import OpenAI
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


@dataclass
class Chunk:
    text: str
    source: str


class RAGIndex:
    def __init__(
        self, embeddings: np.ndarray, chunks: List[Chunk], index, model_name: str
    ) -> None:
        self.embeddings = embeddings
        self.chunks = chunks
        self.index = index
        self.model_name = model_name


def load_pdfs(files: Iterable[bytes], names: Iterable[str]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for file_bytes, name in zip(files, names):
        reader = PdfReader(BytesIO(file_bytes))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        for chunk in split_text(text, chunk_size=900, overlap=150):
            if chunk.strip():
                chunks.append(Chunk(text=chunk, source=name))
    return chunks


def split_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    words = text.split()
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def build_index(chunks: List[Chunk], model_name: str = "all-MiniLM-L6-v2") -> RAGIndex:
    if not chunks:
        raise ValueError("No text chunks found. Upload readable PDFs.")

    model = SentenceTransformer(model_name)
    texts = [chunk.text for chunk in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True)
    embeddings_np = np.array(embeddings, dtype="float32")

    import faiss

    index = faiss.IndexFlatIP(embeddings_np.shape[1])
    index.add(embeddings_np)
    return RAGIndex(
        embeddings=embeddings_np, chunks=chunks, index=index, model_name=model_name
    )


def retrieve(index: RAGIndex, query: str, top_k: int = 4) -> List[Chunk]:
    model = SentenceTransformer(index.model_name)
    query_embedding = model.encode([query], normalize_embeddings=True)
    scores, indices = index.index.search(query_embedding.astype("float32"), top_k)
    results: List[Chunk] = []
    for idx in indices[0]:
        if idx == -1:
            continue
        results.append(index.chunks[idx])
    return results


def answer_with_context(query: str, contexts: List[Chunk]) -> Tuple[str, List[Chunk]]:
    if not contexts:
        return "I couldn't find relevant context in your documents.", []

    context_text = "\n\n".join(
        f"Source: {chunk.source}\n{chunk.text}" for chunk in contexts
    )

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a college document Q&A assistant. Answer strictly using the "
                    "provided context. If the answer is not contained in the context, say "
                    "you don't know and ask the user to upload more material."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {query}\n\nContext:\n{context_text}",
            },
        ],
        temperature=0.2,
    )

    answer = response.choices[0].message.content or ""
    return answer.strip(), contexts
