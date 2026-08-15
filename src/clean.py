"""
Stage 1b — Clean raw texts and build the sentence-chunked master corpus.

Reads every .txt in config.RAW_DIR, strips Project Gutenberg boilerplate,
normalizes punctuation/whitespace, writes cleaned files to config.CLEANED_DIR,
then sentence-tokenizes everything into overlapping chunks and saves the
result as both Parquet and CSV.

Usage:
    python src/clean.py
    python src/clean.py --raw-dir ./VictorianGPT/raw --cleaned-dir ./VictorianGPT/cleaned
"""

import argparse
import re
from pathlib import Path

import nltk
import pandas as pd
from nltk.tokenize import sent_tokenize

from config import (
    CLEANED_DIR,
    MASTER_CSV,
    MASTER_PARQUET,
    RAW_DIR,
    SENTENCE_CHUNK_OVERLAP,
    SENTENCE_CHUNK_SIZE,
)

GUTENBERG_START_MARKER = "*** START"
GUTENBERG_END_MARKER = "*** END"


def ensure_nltk_resources() -> None:
    """Download NLTK's sentence tokenizer models if not already present."""
    for resource in ("tokenizers/punkt", "tokenizers/punkt_tab"):
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource.split("/")[-1])


def clean_text(text: str) -> str:
    """Strip Gutenberg boilerplate and normalize punctuation/whitespace."""
    start = text.find(GUTENBERG_START_MARKER)
    end = text.find(GUTENBERG_END_MARKER)

    if start != -1 and end != -1:
        start_of_line = text.find("\n", start)
        text = text[start_of_line:end] if start_of_line != -1 else text[start:end]

    text = re.sub(r"[“”]", '"', text)
    text = re.sub(r"[‘’]", "'", text)
    text = re.sub(r"—|--", " - ", text)
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_corpus(raw_dir: Path, cleaned_dir: Path) -> None:
    """Clean every .txt file in raw_dir and write results to cleaned_dir."""
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    print("Commencing the cleansing of the archives...\n")

    txt_files = sorted(p for p in raw_dir.iterdir() if p.suffix == ".txt")
    for path in txt_files:
        text = path.read_text(encoding="utf-8")
        cleaned = clean_text(text)
        (cleaned_dir / path.name).write_text(cleaned, encoding="utf-8")

    print(f"Cleaned {len(txt_files)} files into {cleaned_dir}/")


def chunk_corpus(
    cleaned_dir: Path,
    chunk_size: int = SENTENCE_CHUNK_SIZE,
    overlap: int = SENTENCE_CHUNK_OVERLAP,
) -> list[str]:
    """Sentence-tokenize every cleaned file and build overlapping chunks."""
    ensure_nltk_resources()
    chunks: list[str] = []

    txt_files = sorted(p for p in cleaned_dir.iterdir() if p.suffix == ".txt")
    for path in txt_files:
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            continue

        sentences = sent_tokenize(text)
        step = max(chunk_size - overlap, 1)
        for i in range(0, len(sentences), step):
            window = sentences[i : i + chunk_size]
            if window:
                chunks.append(" ".join(window))

    return chunks


def save_master_corpus(chunks: list[str], parquet_path: Path, csv_path: Path) -> pd.DataFrame:
    """Deduplicate chunks and persist to Parquet + CSV."""
    print(f"Binding the master ledger from {len(chunks)} fragments...")

    df = pd.DataFrame({"text": chunks})
    initial_count = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"Purged {initial_count - len(df)} duplicate fragments. {len(df)} chunks remain.")

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False, escapechar="\\")

    print(f"Saved: {parquet_path}")
    print(f"Saved: {csv_path}")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean and chunk the raw VictorianGPT corpus.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--cleaned-dir", type=Path, default=CLEANED_DIR)
    parser.add_argument("--parquet-out", type=Path, default=MASTER_PARQUET)
    parser.add_argument("--csv-out", type=Path, default=MASTER_CSV)
    parser.add_argument("--chunk-size", type=int, default=SENTENCE_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=SENTENCE_CHUNK_OVERLAP)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    clean_corpus(args.raw_dir, args.cleaned_dir)
    chunks = chunk_corpus(args.cleaned_dir, args.chunk_size, args.overlap)
    save_master_corpus(chunks, args.parquet_out, args.csv_out)
