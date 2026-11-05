import argparse
import sys
from pathlib import Path

from . import config
from .pipeline import ask, build_index


def main():
    parser = argparse.ArgumentParser(
        prog="docqa",
        description="Ask questions about a PDF using a fully local RAG pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser(
        "index", help="Read a PDF and build the vector database"
    )
    index_parser.add_argument("pdf", help="Path to the PDF file")

    ask_parser = subparsers.add_parser(
        "ask", help="Ask a question about the indexed PDF"
    )
    ask_parser.add_argument("question", help="Your question, in quotes")

    args = parser.parse_args()

    if args.command == "index":
        n_pages, n_chunks = build_index(args.pdf)
        print(f"Indexed {n_pages} pages as {n_chunks} chunks into '{config.DB_DIR}/'")

    elif args.command == "ask":
        if not Path(config.DB_DIR).exists():
            print("No index found. Run first: python -m docqa index <your.pdf>")
            sys.exit(1)
        answer, sources = ask(args.question)
        print(answer)
        if sources:
            print("\nSources:")
            for page, score in sources:
                print(f"  - page {page} (distance {score:.3f})")


if __name__ == "__main__":
    main()
