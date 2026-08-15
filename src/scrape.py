"""
Stage 1a — Download the source corpus.

Downloads 40 curated 19th-century / Gothic novels from Project Gutenberg into
config.RAW_DIR. Safe to re-run: files that already exist are skipped.

Usage:
    python src/scrape.py
    python src/scrape.py --output-dir ./VictorianGPT/raw
"""

import argparse
import time
from pathlib import Path

import requests

from config import RAW_DIR

BOOKS = {
    # The Original Foundation
    "great_expectations": "https://www.gutenberg.org/cache/epub/1400/pg1400.txt",
    "jane_eyre": "https://www.gutenberg.org/cache/epub/1260/pg1260.txt",
    "dracula": "https://www.gutenberg.org/cache/epub/345/pg345.txt",
    "dorian_gray": "https://www.gutenberg.org/cache/epub/174/pg174.txt",
    # The Dickens Collection
    "oliver_twist": "https://www.gutenberg.org/cache/epub/730/pg730.txt",
    "a_tale_of_two_cities": "https://www.gutenberg.org/cache/epub/98/pg98.txt",
    "david_copperfield": "https://www.gutenberg.org/cache/epub/766/pg766.txt",
    "bleak_house": "https://www.gutenberg.org/cache/epub/1023/pg1023.txt",
    "hard_times": "https://www.gutenberg.org/cache/epub/786/pg786.txt",
    # The Brontë Sisters
    "wuthering_heights": "https://www.gutenberg.org/cache/epub/768/pg768.txt",
    "the_tenant_of_wildfell_hall": "https://www.gutenberg.org/cache/epub/969/pg969.txt",
    "villette": "https://www.gutenberg.org/cache/epub/3154/pg3154.txt",
    # The Gothic & Macabre
    "frankenstein": "https://www.gutenberg.org/cache/epub/84/pg84.txt",
    "dr_jekyll_and_mr_hyde": "https://www.gutenberg.org/cache/epub/42/pg42.txt",
    "the_turn_of_the_screw": "https://www.gutenberg.org/cache/epub/209/pg209.txt",
    "carmilla": "https://www.gutenberg.org/cache/epub/10007/pg10007.txt",
    "the_picture_of_dorian_gray_13_ch": "https://www.gutenberg.org/cache/epub/4058/pg4058.txt",
    # Arthur Conan Doyle (Sherlock Holmes)
    "the_hound_of_the_baskervilles": "https://www.gutenberg.org/cache/epub/2852/pg2852.txt",
    "a_study_in_scarlet": "https://www.gutenberg.org/cache/epub/244/pg244.txt",
    "the_sign_of_the_four": "https://www.gutenberg.org/cache/epub/2097/pg2097.txt",
    "adventures_of_sherlock_holmes": "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",
    # H.G. Wells
    "the_time_machine": "https://www.gutenberg.org/cache/epub/35/pg35.txt",
    "the_war_of_the_worlds": "https://www.gutenberg.org/cache/epub/36/pg36.txt",
    "the_invisible_man": "https://www.gutenberg.org/cache/epub/5230/pg5230.txt",
    "the_island_of_dr_moreau": "https://www.gutenberg.org/cache/epub/159/pg159.txt",
    # Thomas Hardy
    "tess_of_the_durbervilles": "https://www.gutenberg.org/cache/epub/110/pg110.txt",
    "far_from_the_madding_crowd": "https://www.gutenberg.org/cache/epub/107/pg107.txt",
    "jude_the_obscure": "https://www.gutenberg.org/cache/epub/153/pg153.txt",
    # Austen (Regency, but highly compatible dialogue)
    "pride_and_prejudice": "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
    "emma": "https://www.gutenberg.org/cache/epub/158/pg158.txt",
    "sense_and_sensibility": "https://www.gutenberg.org/cache/epub/161/pg161.txt",
    # George Eliot & Elizabeth Gaskell
    "middlemarch": "https://www.gutenberg.org/cache/epub/145/pg145.txt",
    "silas_marner": "https://www.gutenberg.org/cache/epub/550/pg550.txt",
    "north_and_south": "https://www.gutenberg.org/cache/epub/4276/pg4276.txt",
    # Wilkie Collins (Pioneers of mystery)
    "the_woman_in_white": "https://www.gutenberg.org/cache/epub/583/pg583.txt",
    "the_moonstone": "https://www.gutenberg.org/cache/epub/155/pg155.txt",
    # Other Essential 19th-Century Works
    "alice_in_wonderland": "https://www.gutenberg.org/cache/epub/11/pg11.txt",
    "the_scarlet_letter": "https://www.gutenberg.org/cache/epub/253/pg253.txt",
    "moby_dick": "https://www.gutenberg.org/cache/epub/2701/pg2701.txt",
    "les_miserables": "https://www.gutenberg.org/cache/epub/135/pg135.txt",
}


def download_books(output_dir: Path, books: dict = BOOKS, delay: float = 1.5) -> None:
    """Download each book to `output_dir/<name>.txt`, skipping existing files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Preparing to requisition {len(books)} manuscripts...\n")

    for name, url in books.items():
        file_path = output_dir / f"{name}.txt"

        if file_path.exists():
            print(f"[{name}] already rests in the archives. Skipping.")
            continue

        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            file_path.write_text(response.text, encoding="utf-8")
            print(f"Successfully secured: {name}")
            time.sleep(delay)  # be polite to Gutenberg's servers
        except requests.exceptions.RequestException as e:
            print(f"Failed to procure [{name}]: {e}")

    print(f"\nProcurement complete. Raw texts now reside in {output_dir}/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the VictorianGPT source corpus.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RAW_DIR,
        help=f"Directory to save raw .txt files into (default: {RAW_DIR})",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="Seconds to sleep between downloads (default: 1.5)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_books(args.output_dir, delay=args.delay)
