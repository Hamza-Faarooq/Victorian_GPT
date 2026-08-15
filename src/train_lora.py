"""
Stage 3 — LoRA fine-tune Qwen2.5-3B-Instruct on the Victorian dialogue dataset.

Uses Unsloth for 4-bit loading and efficient LoRA training, and TRL's
SFTTrainer for the training loop. Designed to run on a single free-tier
Colab T4 GPU, but works on any CUDA GPU.

Usage:
    python src/train_lora.py
    python src/train_lora.py --dataset ./VictorianGPT/dialogues/victorian_dataset_master.json \
                              --output-dir ./VictorianGPT/Victorian_Qwen_Final_Adapter \
                              --epochs 1
"""

import argparse
from pathlib import Path

from config import ADAPTER_DIR, BASE_MODEL, CHECKPOINT_DIR, MASTER_DIALOGUE_JSON

CHAT_TEMPLATE = "<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n{output_text}<|im_end|>"


def format_dataset(dataset):
    """Format input/output pairs into Qwen's ChatML-style template."""

    def _format(examples):
        texts = [
            CHAT_TEMPLATE.format(input_text=i, output_text=o)
            for i, o in zip(examples["input"], examples["output"])
        ]
        return {"text": texts}

    return dataset.map(_format, batched=True)


def train(
    dataset_path: Path,
    output_dir: Path,
    checkpoint_dir: Path,
    base_model: str = BASE_MODEL,
    max_seq_length: int = 2048,
    lora_r: int = 16,
    lora_alpha: int = 16,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    num_train_epochs: int = 1,
    learning_rate: float = 2e-4,
    warmup_steps: int = 50,
) -> None:
    # Heavy, GPU-only imports kept local so this module can still be
    # inspected / imported for its config on CPU-only machines.
    import torch
    from datasets import load_dataset
    from transformers import TrainingArguments
    from trl import SFTTrainer
    from unsloth import FastLanguageModel

    print(f"Loading base model: {base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=None,  # auto-detect
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    print(f"Loading dataset: {dataset_path}")
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    dataset = format_dataset(dataset)
    print(f"Prepared {len(dataset)} training examples.")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=str(checkpoint_dir),
            save_strategy="steps",
            save_steps=100,
            save_total_limit=2,
        ),
    )

    print("Starting training...")
    trainer.train()

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"\nTraining complete. Adapter saved to:\n{output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tune Qwen2.5-3B-Instruct on VictorianGPT data.")
    parser.add_argument("--dataset", type=Path, default=MASTER_DIALOGUE_JSON)
    parser.add_argument("--output-dir", type=Path, default=ADAPTER_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--base-model", type=str, default=BASE_MODEL)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        base_model=args.base_model,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
    )
