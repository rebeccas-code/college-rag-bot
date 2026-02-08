import os
from typing import List

import streamlit as st

from rag import RAGIndex, answer_with_context, build_index, load_pdfs, retrieve

st.set_page_config(page_title="College Document Q&A Bot", layout="wide")

st.title("College Document Q&A Bot")
st.caption("RAG-powered assistant for syllabus, VTU question papers, and lecture notes.")

if "rag_index" not in st.session_state:
    st.session_state.rag_index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

st.sidebar.header("Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs", type=["pdf"], accept_multiple_files=True
)

build_clicked = st.sidebar.button("Build Index")

if build_clicked:
    if not uploaded_files:
        st.sidebar.error("Please upload at least one PDF.")
    else:
        with st.spinner("Reading PDFs and building index..."):
            file_bytes = [file.getvalue() for file in uploaded_files]
            file_names = [file.name for file in uploaded_files]
            chunks = load_pdfs(file_bytes, file_names)
            st.session_state.chunks = chunks
            st.session_state.rag_index = build_index(chunks)
        st.sidebar.success("Index built! You can now ask questions.")

if not os.getenv("OPENAI_API_KEY"):
    st.warning(
        "OPENAI_API_KEY is not set. Add it to your environment to enable answer generation."
    )

st.subheader("Ask a question")
question = st.text_input(
    "Try: 'Which modules are important for IAT?'",
    placeholder="Ask about modules, repeated questions, or explain a topic...",
)

if st.button("Get Answer"):
    if not question.strip():
        st.error("Please enter a question.")
    elif not isinstance(st.session_state.rag_index, RAGIndex):
        st.error("Please upload PDFs and build the index first.")
    else:
        with st.spinner("Retrieving context and generating answer..."):
            contexts = retrieve(st.session_state.rag_index, question)
            answer, used_chunks = answer_with_context(question, contexts)

        st.markdown("### Answer")
        st.write(answer)

        if used_chunks:
            st.markdown("### Sources")
            for idx, chunk in enumerate(used_chunks, start=1):
                with st.expander(f"Source {idx}: {chunk.source}"):
                    st.write(chunk.text)
