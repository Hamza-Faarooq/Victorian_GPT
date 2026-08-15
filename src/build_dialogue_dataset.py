"""
Stage 2 — Build the supervised fine-tuning dialogue dataset.

Combines two sources into a single master dataset:
  1. "Authentic" pairs: consecutive quoted lines of dialogue extracted
     directly from the cleaned novels.
  2. "Synthetic" pairs: modern, first-person emotional prompts translated
     into Victorian-register responses by Qwen2.5-3B-Instruct.

Usage:
    python src/build_dialogue_dataset.py
    python src/build_dialogue_dataset.py --skip-synthetic   # authentic pairs only, no GPU needed
"""

import argparse
import json
import re
from pathlib import Path

from config import (
    CLEANED_DIR,
    DATASET_GENERATION_PROMPT_TEMPLATE,
    GENERATOR_MODEL,
    MASTER_DIALOGUE_JSON,
)

EMOTIONS = ["sad", "happy", "lonely", "anxious", "stressed", "worried", "confused", "excited", "angry"]
SITUATIONS = [
    "after exams", "after an interview", "after bad news", "while studying",
    "after losing something", "during college", "after an argument", "after a long day",
]
ACTIONS = ["I feel", "I became", "I am", "I suddenly feel", "I think I am", "I am feeling"]


def extract_authentic_pairs(cleaned_dir: Path) -> list[dict]:
    """Pair up consecutive quoted lines of dialogue from each cleaned novel."""
    print("Commencing extraction from authentic manuscripts...")

    authentic_pairs: list[dict] = []
    total_quotes = 0

    for path in sorted(p for p in cleaned_dir.iterdir() if p.suffix == ".txt"):
        text = path.read_text(encoding="utf-8")
        raw_dialogues = re.findall(r'["“](.*?)["”]', text, re.DOTALL)

        book_dialogues = []
        for d in raw_dialogues:
            clean_quote = re.sub(r"\s+", " ", d.replace("\n", " ")).strip()
            if len(clean_quote) > 5:
                book_dialogues.append(clean_quote)

        total_quotes += len(book_dialogues)

        for i in range(len(book_dialogues) - 1):
            input_text, output_text = book_dialogues[i], book_dialogues[i + 1]
            if len(input_text) > 15 and len(output_text) > 15:
                authentic_pairs.append(
                    {"input": input_text, "output": output_text, "source": "authentic_novel"}
                )

    print(f"Total individual quotes extracted: {total_quotes}")
    print(f"Total authentic conversational pairs created: {len(authentic_pairs)}")
    return authentic_pairs


def build_modern_prompts() -> list[str]:
    """Build the cross-product of actions x emotions x situations."""
    prompts = [
        f"{a} {e} {s}" for a in ACTIONS for e in EMOTIONS for s in SITUATIONS
    ]
    print(f"Generated {len(prompts)} modern conversational prompts.")
    return prompts


def generate_synthetic_pairs(
    modern_inputs: list[str],
    model_name: str = GENERATOR_MODEL,
    batch_size: int = 8,
    max_new_tokens: int = 50,
) -> list[dict]:
    """Translate modern prompts into Victorian-register responses via an LLM.

    Requires a GPU-backed environment with `transformers` and `accelerate`
    installed. This is the most expensive step in the dataset pipeline.
    """
    from transformers import pipeline  # local import: heavy, optional dependency

    print(f"Loading {model_name} for synthetic generation...")
    generator = pipeline(
        "text-generation",
        model=model_name,
        device_map="auto",
        batch_size=batch_size,
    )
    print("Model loaded.")

    prompts = [DATASET_GENERATION_PROMPT_TEMPLATE.format(text=text) for text in modern_inputs]

    print("Generating synthetic Victorian responses...")
    outputs = generator(
        prompts,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
    )

    synthetic_pairs = []
    for original_text, out in zip(modern_inputs, outputs):
        raw_response = out[0]["generated_text"]
        if "Victorian response:" in raw_response:
            clean_response = raw_response.split("Victorian response:")[-1].strip()
        else:
            clean_response = raw_response.strip()

        synthetic_pairs.append(
            {"input": original_text, "output": clean_response, "source": "synthetic_qwen"}
        )

    print(f"Successfully generated {len(synthetic_pairs)} synthetic pairs.")
    return synthetic_pairs


def save_dataset(pairs: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=4)
    print(f"\nSaved master dataset ({len(pairs)} pairs) to:\n{output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the VictorianGPT dialogue dataset.")
    parser.add_argument("--cleaned-dir", type=Path, default=CLEANED_DIR)
    parser.add_argument("--output", type=Path, default=MASTER_DIALOGUE_JSON)
    parser.add_argument("--model-name", type=str, default=GENERATOR_MODEL)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--skip-synthetic",
        action="store_true",
        help="Skip LLM-based synthetic pair generation (authentic pairs only, no GPU required).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    authentic_pairs = extract_authentic_pairs(args.cleaned_dir)

    if args.skip_synthetic:
        synthetic_pairs = []
    else:
        modern_inputs = build_modern_prompts()
        synthetic_pairs = generate_synthetic_pairs(
            modern_inputs, model_name=args.model_name, batch_size=args.batch_size
        )

    master_dataset = authentic_pairs + synthetic_pairs
    print(f"\nAuthentic Pairs: {len(authentic_pairs)}")
    print(f"Synthetic Pairs: {len(synthetic_pairs)}")
    print(f"Total Master Dataset Size: {len(master_dataset)}")

    save_dataset(master_dataset, args.output)
