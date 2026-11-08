import argparse
import sys
from pathlib import Path

from . import config
from .pipeline import ask, build_index


def main():
    parser = argparse.ArgumentParser(
        prog="docqa",
        description="Ask questions about PDFs using a fully local RAG pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser(
        "index", help="Read PDFs and build the vector database"
    )
    index_parser.add_argument("path", help="A PDF file, or a folder of PDFs")

    ask_parser = subparsers.add_parser(
        "ask", help="Ask a question about the indexed PDFs"
    )
    ask_parser.add_argument("question", help="Your question, in quotes")

    args = parser.parse_args()

    if args.command == "index":
        try:
            n_files, n_pages, n_chunks = build_index(args.path)
        except FileNotFoundError as error:
            print(error)
            sys.exit(1)
        print(
            f"Indexed {n_files} file(s), {n_pages} pages, {n_chunks} chunks "
            f"into '{config.DB_DIR}/'"
        )

    elif args.command == "ask":
        if not Path(config.DB_DIR).exists():
            print("No index found. Run first: python -m docqa index <file or folder>")
            sys.exit(1)
        answer, sources = ask(args.question)
        print(answer)
        if sources:
            print("\nSources:")
            for source, page, score in sources:
                print(f"  - {source}, page {page} (distance {score:.3f})")


if __name__ == "__main__":
    main()
