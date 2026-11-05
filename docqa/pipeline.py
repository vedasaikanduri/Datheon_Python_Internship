import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config
from .loader import load_pdf


def _embeddings():
    return OllamaEmbeddings(model=config.EMBED_MODEL)


def build_index(pdf_path):
    """Load a PDF, chunk it, embed the chunks and save them to ChromaDB."""
    pages = load_pdf(pdf_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(pages)

    if Path(config.DB_DIR).exists():
        shutil.rmtree(config.DB_DIR)

    Chroma.from_documents(
        documents=chunks,
        embedding=_embeddings(),
        persist_directory=config.DB_DIR,
    )
    return len(pages), len(chunks)


def ask(question):
    """Answer a question from the indexed PDF. Returns (answer, sources)."""
    db = Chroma(persist_directory=config.DB_DIR, embedding_function=_embeddings())
    results = db.similarity_search_with_score(question, k=config.TOP_K)

    relevant = [(doc, score) for doc, score in results if score <= config.MAX_DISTANCE]
    if not relevant:
        return config.NOT_FOUND_MESSAGE, []

    context = "\n\n".join(
        f"[Page {doc.metadata['page']}]\n{doc.page_content}" for doc, _ in relevant
    )

    prompt = ChatPromptTemplate.from_messages(
        [("system", config.SYSTEM_PROMPT), ("human", "{question}")]
    )
    llm = ChatOllama(model=config.CHAT_MODEL, temperature=0)
    response = (prompt | llm).invoke({"context": context, "question": question})

    sources = [(doc.metadata["page"], score) for doc, score in relevant]
    return response.content, sources
