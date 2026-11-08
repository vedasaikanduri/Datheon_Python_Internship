import shutil
from pathlib import Path

import streamlit as st

from docqa import config
from docqa.pipeline import ask, build_index

UPLOAD_DIR = Path("uploads")

st.set_page_config(page_title="docqa", page_icon="📚")
st.title("📚 docqa")
st.caption(
    "Ask questions about your PDFs. Runs fully locally with Ollama, "
    "so nothing leaves your computer."
)

if "messages" not in st.session_state:
    st.session_state.messages = []


def unique_sources(sources):
    """One entry per (file, page), keeping the closest distance, closest first."""
    best = {}
    for source, page, score in sources:
        key = (source, page)
        if key not in best or score < best[key]:
            best[key] = score
    return sorted(best.items(), key=lambda item: item[1])


def show_sources(sources):
    with st.expander("Retrieved from"):
        for (source, page), score in unique_sources(sources):
            st.write(f"{source}, page {page} (distance {score:.3f})")


# ---- Sidebar: upload and index documents ----
with st.sidebar:
    st.header("Documents")
    uploaded = st.file_uploader(
        "Upload PDFs", type="pdf", accept_multiple_files=True
    )

    if st.button("Index documents", disabled=not uploaded):
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
        UPLOAD_DIR.mkdir()
        for file in uploaded:
            (UPLOAD_DIR / Path(file.name).name).write_bytes(file.getvalue())
        with st.spinner("Reading, chunking and embedding..."):
            try:
                n_files, n_pages, n_chunks = build_index(str(UPLOAD_DIR))
                st.session_state.messages = []
                st.success(
                    f"Indexed {n_files} file(s), {n_pages} pages, {n_chunks} chunks"
                )
            except Exception as error:
                st.error(f"Indexing failed: {error}")

    if Path(config.DB_DIR).exists():
        st.caption("An index is ready.")
    else:
        st.caption("No index yet. Upload PDFs and click Index documents.")

    if st.button("Clear chat"):
        st.session_state.messages = []

# ---- Chat history ----
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            show_sources(message["sources"])

# ---- New question ----
question = st.chat_input("Ask a question about your documents")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not Path(config.DB_DIR).exists():
            answer, sources = "Please upload and index some PDFs first.", []
        else:
            with st.spinner("Thinking..."):
                try:
                    answer, sources = ask(question)
                except Exception as error:
                    answer = (
                        f"Something went wrong: {error}\n\n"
                        "Is Ollama running? Look for the llama icon in your menu bar."
                    )
                    sources = []
        st.markdown(answer)
        if sources:
            show_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
