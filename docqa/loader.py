from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader


def load_pdf(path):
    """Read a PDF and return one Document per page."""
    reader = PdfReader(path)
    source = Path(path).name
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(
            Document(
                page_content=text,
                metadata={"source": source, "page": page_number},
            )
        )
    return pages
