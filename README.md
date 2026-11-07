# docqa: Local PDF Question Answering with RAG

A command-line tool that answers questions about a PDF using Retrieval-Augmented Generation (RAG), running **entirely on your own machine**. No API keys, no cloud services, and your documents never leave your computer.

## Features

- **Fully local:** Ollama runs both the embedding model and the chat model
- **Page citations:** every answer says which pages it came from
- **Stays on topic:** a distance cutoff plus a strict prompt make it refuse questions the document can't answer
- **Persistent index:** index a PDF once, ask as many questions as you like
- **Clean structure:** all settings live in one config file

## How it works

```
INDEXING   PDF -> pypdf (one Document per page)
               -> RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
               -> nomic-embed-text (vectors)
               -> ChromaDB (saved to disk)

ASKING     question -> embed -> 4 nearest chunks from ChromaDB
                    -> drop chunks above the distance cutoff
                    -> prompt (question + chunks + rules)
                    -> gemma3:12b -> answer with page citations
```

If no chunk is close enough to the question, the language model is never called and the tool says it couldn't find the answer.

## Prerequisites

- Python 3.10 or newer (developed and tested on 3.12)
- [Ollama](https://ollama.com/download) installed and running

## Setup

```bash
git clone https://github.com/vedasaikanduri/doc-qa-rag.git
cd doc-qa-rag

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

ollama pull nomic-embed-text
ollama pull gemma3:12b
```

## Usage

Put a text-based PDF in the `data/` folder, then index it and ask questions:

```bash
python -m docqa index data/your-file.pdf
python -m docqa ask "your question here"
```

Questions the document can't answer get a refusal instead of a made-up reply:

```
$ python -m docqa ask "who won the world cup?"
I couldn't find that in the document.
```

Run `python -m docqa --help` for all commands.

## Configuration

Edit `docqa/config.py`:

| Setting | Meaning |
|---|---|
| `CHAT_MODEL` / `EMBED_MODEL` | Which Ollama models to use |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | How the text is split (characters) |
| `TOP_K` | How many chunks to retrieve per question |
| `MAX_DISTANCE` | Cutoff above which chunks count as irrelevant (lower = closer match) |

## Project structure

```
docqa/
  __main__.py   command-line interface
  config.py     settings and the system prompt
  loader.py     PDF to per-page Documents
  pipeline.py   indexing and question answering
```

## Evaluation

`docqa/evaluate.py` checks retrieval against a hand-written test set (`eval/questions.json`): 19 questions whose answers are on known pages, and 8 off-topic questions the document does not answer (3 clearly unrelated, 5 nearby biology topics that sound relevant but are not covered). The test set was written for one document (AQA A-level Biology notes, not included in this repo), so the page numbers only match that PDF. To reuse the script, replace the questions with ones for your own document.

```bash
python -m docqa.evaluate
```

Results with `chunk_size=1000`, `chunk_overlap=200`, `nomic-embed-text` and the top 4 chunks:

- The correct page was in the top 4 for **19/19** answerable questions.
- Cutoff sweep (answerable questions kept / off-topic questions wrongly passed):

| Cutoff | Answerable kept | Off-topic passed |
|---|---|---|
| 0.6 | 14/19 | 0/8 |
| 0.7 | 17/19 | 0/8 |
| **0.8** | **19/19** | **1/8** |
| 0.9 | 19/19 | 5/8 |
| 1.0 | 19/19 | 5/8 |
| 1.1 | 19/19 | 6/8 |
| 1.2 | 19/19 | 8/8 |

`MAX_DISTANCE` is set to 0.8. The one off-topic question that still passes the cutoff (kidney filtration, distance 0.776) is rejected by the language model: in an end-to-end test it replied that it could not find the answer. The cutoff catches clearly unrelated questions, and the strict prompt catches the borderline ones.

Caveat: the cutoff was tuned on the same questions it was measured on, and the margin is small, so treat 0.8 as a good starting point rather than a proven value.

## Limitations

- One PDF at a time (re-indexing replaces the previous index)
- Text-based PDFs only; scanned documents would need OCR
- `MAX_DISTANCE` was tuned on a small test set of 27 questions written by me, so it may not generalise to other documents
- Text is split per page, so an idea that runs across a page break is cut there
- The *Sources* list shows the chunks that were retrieved, not necessarily every one the model used

## Roadmap

- Evaluation script to measure retrieval quality
- Support for multiple PDFs
- Web interface

## Note on documents

PDFs in `data/` are git-ignored. Bring your own, and respect the copyright of anything you index.

## Licence

MIT. See [LICENSE](LICENSE).
