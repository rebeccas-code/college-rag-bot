# College Document Q&A Bot

A lightweight RAG-powered assistant for college documents. Upload PDFs (syllabus, VTU question papers, lecture notes), build a local vector index, and ask questions grounded **only** in your materials.

## Features
- **Upload** multiple PDFs (syllabus, notes, VTU question papers).
- **Ask** focused questions like:
  - “Which modules are important for IAT?”
  - “List repeated VTU questions from Module 3”
  - “Explain this topic based on my notes only”
- **Answers are grounded** in retrieved chunks with source snippets.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# set your OpenAI API key (required for answer generation)
export OPENAI_API_KEY=your_key_here

streamlit run app.py
```

## Notes
- The app uses sentence-transformer embeddings and FAISS for similarity search.
- LLM responses are strictly limited to retrieved context; if the answer is not present, the bot will say it doesn’t know.

## Project Structure
- `app.py` – Streamlit UI + session state.
- `rag.py` – PDF parsing, chunking, embedding, retrieval, and LLM answer generation.
