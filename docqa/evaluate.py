import json

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from . import config

EVAL_FILE = "eval/questions.json"
THRESHOLDS = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]


def main():
    with open(EVAL_FILE) as f:
        items = json.load(f)

    db = Chroma(
        persist_directory=config.DB_DIR,
        embedding_function=OllamaEmbeddings(model=config.EMBED_MODEL),
    )

    best_correct = []  # answerable questions: best distance among chunks from a correct page
    best_any = []      # unanswerable questions: best distance among all chunks

    for item in items:
        results = db.similarity_search_with_score(item["question"], k=config.TOP_K)
        retrieved = [(doc.metadata["page"], score) for doc, score in results]

        if item["pages"]:
            correct = [s for page, s in retrieved if page in item["pages"]]
            best = min(correct) if correct else None
            best_correct.append(best)
            status = "HIT " if best is not None else "MISS"
            shown = f"{best:.3f}" if best is not None else "  -  "
            print(f"{status} | pages {item['pages']} | best correct distance {shown} | {item['question']}")
        else:
            best = min(score for _, score in retrieved)
            best_any.append(best)
            print(f"OFF  | closest chunk distance {best:.3f} | {item['question']}")

    hits = sum(1 for b in best_correct if b is not None)
    print(f"\nRetrieval: the correct page was in the top {config.TOP_K} for {hits}/{len(best_correct)} answerable questions")

    print("\nCutoff sweep (a question 'passes' if a qualifying chunk is within the cutoff):")
    print("cutoff | answerable kept | off-topic wrongly passed")
    for t in THRESHOLDS:
        kept = sum(1 for b in best_correct if b is not None and b <= t)
        wrong = sum(1 for b in best_any if b <= t)
        print(f"{t:>6} | {kept}/{len(best_correct):<14} | {wrong}/{len(best_any)}")
    print(f"\nCurrent MAX_DISTANCE = {config.MAX_DISTANCE}")


if __name__ == "__main__":
    main()
