"""
Central configuration for the VictorianGPT pipeline.

All scripts import their default paths from here so the whole project can be
re-pointed (e.g. from a local checkout to a mounted Google Drive path in
Colab) by editing a single file, or by overriding with environment variables.

Paths mirror the repository layout described in README.md:

    data/raw/                              (gitignored)
    data/cleaned/                          (gitignored)
    data/chunks_master.parquet / .csv
    data/dialogues/victorian_dataset_master.json
    models/Victorian_Qwen_Final_Adapter/   (gitignored)
    vectorstore/chroma_db/                 (gitignored)

Every path can be overridden with an environment variable of the same name,
e.g.:
    export VICTORIANGPT_ROOT=/content/drive/MyDrive/VictorianGPT
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Root project directory. Defaults to the repository root (this file's
# grandparent directory). Override with the VICTORIANGPT_ROOT env var to
# point at a different location, e.g. a mounted Google Drive path in Colab.
# ---------------------------------------------------------------------------
ROOT = Path(os.environ.get("VICTORIANGPT_ROOT", Path(__file__).resolve().parent.parent))

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CLEANED_DIR = DATA_DIR / "cleaned"
DIALOGUE_DIR = DATA_DIR / "dialogues"

MODELS_DIR = ROOT / "models"
ADAPTER_DIR = MODELS_DIR / "Victorian_Qwen_Final_Adapter"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"

VECTORSTORE_DIR = ROOT / "vectorstore" / "chroma_db"

MASTER_PARQUET = DATA_DIR / "chunks_master.parquet"
MASTER_CSV = DATA_DIR / "chunks_master.csv"
MASTER_DIALOGUE_JSON = DIALOGUE_DIR / "victorian_dataset_master.json"

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
BASE_MODEL = os.environ.get("VICTORIANGPT_BASE_MODEL", "unsloth/Qwen2.5-3B-Instruct")
GENERATOR_MODEL = os.environ.get("VICTORIANGPT_GENERATOR_MODEL", "Qwen/Qwen2.5-3B-Instruct")
EMBEDDING_MODEL = os.environ.get(
    "VICTORIANGPT_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------------------------------------------------------------------
# Chunking / retrieval parameters
# ---------------------------------------------------------------------------
SENTENCE_CHUNK_SIZE = 6       # sentences per chunk (used in clean.py)
SENTENCE_CHUNK_OVERLAP = 2    # sentence overlap between chunks

RAG_CHUNK_SIZE = 1000         # characters per chunk (used in build_vectorstore.py)
RAG_CHUNK_OVERLAP = 100

RETRIEVAL_SCORE_THRESHOLD = 1.2  # L2 distance; above this = "no relevant lore found"

# ---------------------------------------------------------------------------
# Persona
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an elegant, highly educated, and empathetic gentleman living in "
    "late 19th-century London. "
    "CRITICAL INSTRUCTIONS: "
    "1. Listen carefully to the SPECIFIC subject the user is discussing. "
    "2. If the user expresses general sadness, weariness, or having a 'bad day', "
    "offer poetic, philosophical Victorian sympathy about life's trials "
    "(e.g., comparing their mood to a passing storm or the London fog). "
    "3. ONLY IF the user specifically mentions commerce, jobs, companies, or "
    "interviews, should you frame your advice around 'merchant houses', "
    "'clerks', or 'patrons'. "
    "4. NEVER use the word 'modern', and never break character to explain your "
    "analogies. Speak naturally from your era. "
    "5. Maintain a polite, dignified distance. Do not invent a fictional "
    "lifelong friendship. "
    "6. Respond with exactly 2 to 3 full, eloquent sentences."
)

DATASET_GENERATION_PROMPT_TEMPLATE = """You are converting modern English into natural Victorian English.
Rules:
- Speak like an educated person from late 19th-century England
- Avoid Shakespeare completely
- Never use: thou,thee,thy,dost,hast,methinks,yea
- Keep replies concise (1–3 sentences)
- Do not invent fictional situations
- Do not pretend to physically accompany the user
- Never explain your writing choices
- Never say: "feel free to modify"
- Reply naturally as a chatbot

Modern sentence:
{text}

Victorian response:"""
