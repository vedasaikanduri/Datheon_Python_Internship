import re
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config
from .loader import load_pdf


def _embeddings():
    return OllamaEmbeddings(model=config.EMBED_MODEL)


def _find_pdfs(path):
    """Return a list of PDF paths from a single file or a folder."""
    p = Path(path)
    if p.is_dir():
        pdfs = sorted(p.glob("*.pdf"))
    elif p.is_file():
        pdfs = [p]
    else:
        pdfs = []
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found at '{path}'")
    return pdfs


def _resolve_citations(answer, relevant):
    """Replace passage numbers like [3] with the real (file, page) of that passage."""

    def replace(match):
        numbers = [int(n) for n in re.findall(r"\d+", match.group(1))]
        parts = []
        for n in numbers:
            if 1 <= n <= len(relevant):
                doc = relevant[n - 1][0]
                parts.append(f"{doc.metadata['source']}, page {doc.metadata['page']}")
        if not parts:
            return match.group(0)
        return "(" + "; ".join(parts) + ")"

    return re.sub(r"\[(\d+(?:\s*,\s*\d+)*)\]", replace, answer)


def build_index(path):
    """Index one PDF or every PDF in a folder. Returns (files, pages, chunks)."""
    pdfs = _find_pdfs(path)

    pages = []
    for pdf in pdfs:
        pages.extend(load_pdf(str(pdf)))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(pages)

    # Clear the old index by deleting its collection, not the folder, so this
    # also works inside a long-running program such as the web app
    if Path(config.DB_DIR).exists():
        Chroma(
            persist_directory=config.DB_DIR, embedding_function=_embeddings()
        ).delete_collection()

    Chroma.from_documents(
        documents=chunks,
        embedding=_embeddings(),
        persist_directory=config.DB_DIR,
    )
    return len(pdfs), len(pages), len(chunks)


def ask(question):
    """Answer a question from the indexed PDFs. Returns (answer, sources)."""
    db = Chroma(persist_directory=config.DB_DIR, embedding_function=_embeddings())
    results = db.similarity_search_with_score(question, k=config.TOP_K)

    relevant = [(doc, score) for doc, score in results if score <= config.MAX_DISTANCE]
    if not relevant:
        return config.NOT_FOUND_MESSAGE, []

    context = "\n\n".join(
        f"[{i}]\n{doc.page_content}" for i, (doc, _) in enumerate(relevant, start=1)
    )

    prompt = ChatPromptTemplate.from_messages(
        [("system", config.SYSTEM_PROMPT), ("human", "{question}")]
    )
    llm = ChatOllama(model=config.CHAT_MODEL, temperature=0)
    response = (prompt | llm).invoke({"context": context, "question": question})

    answer = _resolve_citations(response.content, relevant)
    sources = [
        (doc.metadata["source"], doc.metadata["page"], score) for doc, score in relevant
    ]
    return answer, sources
