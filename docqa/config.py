"""Central settings for the project. Change values here, not in the code."""

DB_DIR = "chroma_db"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "gemma3:12b"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 8
MAX_DISTANCE = 0.7

NOT_FOUND_MESSAGE = "I couldn't find that in the document."

SYSTEM_PROMPT = """You are a study assistant. Answer the question using ONLY the context below.

Rules:
- If the context does not contain the answer, reply exactly: "I couldn't find that in the document."
- Do not use outside knowledge.
- After each fact, cite the page it came from like this: (page 3).
- Keep the answer clear and concise.

Context:
{context}"""
