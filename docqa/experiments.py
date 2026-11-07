import json

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config
from .loader import load_pdf

PDF_PATH = "data/sample.pdf"
EVAL_FILE = "eval/questions.json"
CHUNK_SIZES = [500, 1000, 1500]
KS = [1, 2, 4, 6]
MAX_K = max(KS)


def run_setting(pages, answerable, off_topic, chunk_size, embeddings):
    """Build a temporary index for one chunk size and measure it."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_size // 5
    )
    chunks = splitter.split_documents(pages)
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=f"experiment_{chunk_size}",
    )

    hits = {k: 0 for k in KS}
    best_correct = []
    for item in answerable:
        results = db.similarity_search_with_score(item["question"], k=MAX_K)
        for k in KS:
            if any(doc.metadata["page"] in item["pages"] for doc, _ in results[:k]):
                hits[k] += 1
        correct = [s for doc, s in results if doc.metadata["page"] in item["pages"]]
        best_correct.append(min(correct) if correct else float("inf"))

    closest_off = []
    for item in off_topic:
        results = db.similarity_search_with_score(item["question"], k=MAX_K)
        closest_off.append(min(score for _, score in results))

    return {
        "chunks": len(chunks),
        "hits": hits,
        "worst_real": max(best_correct),
        "closest_off": min(closest_off),
    }


def main():
    with open(EVAL_FILE) as f:
        items = json.load(f)
    answerable = [i for i in items if i["pages"]]
    off_topic = [i for i in items if not i["pages"]]
    n = len(answerable)

    pages = load_pdf(PDF_PATH)
    embeddings = OllamaEmbeddings(model=config.EMBED_MODEL)

    print(f"{n} answerable and {len(off_topic)} off-topic questions\n")
    header = f"{'size':>5} {'chunks':>7}  "
    header += "  ".join(f"{'hit@' + str(k):>7}" for k in KS)
    header += f"  {'worst real':>10} {'closest off':>11} {'margin':>7}"
    print(header)

    for size in CHUNK_SIZES:
        r = run_setting(pages, answerable, off_topic, size, embeddings)
        margin = r["closest_off"] - r["worst_real"]
        row = f"{size:>5} {r['chunks']:>7}  "
        row += "  ".join(f"{str(r['hits'][k]) + '/' + str(n):>7}" for k in KS)
        row += f"  {r['worst_real']:>10.3f} {r['closest_off']:>11.3f} {margin:>7.3f}"
        print(row)


if __name__ == "__main__":
    main()
