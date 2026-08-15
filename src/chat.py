"""
Stage 4b — Interactive chat with retrieval-confidence routing.

Loads the fine-tuned LoRA adapter and the persisted ChromaDB vector store,
then runs an interactive loop: each user message is embedded and matched
against the corpus. If a strong match is found, the passage is injected as
grounding context ("Historian Mode"); otherwise the model falls back to a
general empathetic Victorian persona ("Empathy Mode").

Usage:
    python src/chat.py
    python src/chat.py --adapter-dir ./VictorianGPT/Victorian_Qwen_Final_Adapter
"""

import argparse
from pathlib import Path

from config import (
    ADAPTER_DIR,
    BASE_MODEL,
    EMBEDDING_MODEL,
    RETRIEVAL_SCORE_THRESHOLD,
    SYSTEM_PROMPT,
    VECTORSTORE_DIR,
)


def load_model(adapter_dir: Path, base_model: str = BASE_MODEL, max_seq_length: int = 2048):
    from unsloth import FastLanguageModel

    print(f"Loading base model ({base_model}) with adapter from {adapter_dir}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir) if adapter_dir.exists() else base_model,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def load_vectorstore(persist_dir: Path, embedding_model: str = EMBEDDING_MODEL):
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    return Chroma(persist_directory=str(persist_dir), embedding_function=embeddings)


def retrieve_context(vectorstore, query: str, threshold: float = RETRIEVAL_SCORE_THRESHOLD):
    """Return (mode, context) — mode is 'historian' or 'empathy'."""
    results = vectorstore.similarity_search_with_score(query, k=1)
    if not results:
        return "empathy", "No historical context needed. The user requires a sympathetic friend."

    doc, score = results[0]
    print(f"Database match score: {score:.2f} (lower is better)")

    if score > threshold:
        return "empathy", "No historical context needed. The user requires a sympathetic friend."
    return "historian", doc.page_content


def generate_response(model, tokenizer, query: str, context: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context: {context}\n\nUser: {query}"},
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([formatted_prompt], return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        min_new_tokens=30,
        temperature=0.5,
        repetition_penalty=1.1,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # The decoded text includes the prompt; return just the new content.
    return decoded[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):].strip()


def chat_loop(model, tokenizer, vectorstore, threshold: float) -> None:
    print("\nVictorianGPT is ready. Type 'quit' to exit.\n")
    while True:
        query = input("You: ").strip()
        if query.lower() in {"quit", "exit"}:
            print("VictorianGPT: I bid you good day.")
            break
        if not query:
            continue

        mode, context = retrieve_context(vectorstore, query, threshold)
        print(f"[Router] {'Historical lore found' if mode == 'historian' else 'No relevant lore found'} "
              f"-> {mode.title()} Mode")

        response = generate_response(model, tokenizer, query, context)
        print(f"VictorianGPT: {response}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with VictorianGPT.")
    parser.add_argument("--adapter-dir", type=Path, default=ADAPTER_DIR)
    parser.add_argument("--base-model", type=str, default=BASE_MODEL)
    parser.add_argument("--vectorstore-dir", type=Path, default=VECTORSTORE_DIR)
    parser.add_argument("--embedding-model", type=str, default=EMBEDDING_MODEL)
    parser.add_argument("--threshold", type=float, default=RETRIEVAL_SCORE_THRESHOLD)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model, tokenizer = load_model(args.adapter_dir, args.base_model)
    vectorstore = load_vectorstore(args.vectorstore_dir, args.embedding_model)
    chat_loop(model, tokenizer, vectorstore, args.threshold)
