# 🕯️ VictorianGPT

**A fine-tuned, retrieval-augmented chatbot that speaks and reasons like an educated gentleman of late 19th-century England.**

VictorianGPT takes a modern instruction-tuned LLM (Qwen2.5-3B-Instruct), fine-tunes it with LoRA on a custom-built dataset of Victorian-era dialogue, and grounds its answers in a retrieval layer built from 40 curated 19th-century novels — so it doesn't just *sound* Victorian, it can also draw on the actual texts of Dickens, the Brontës, Stoker, Wells, Doyle, Hardy, Austen, and more.

> Ask it about your day, and it answers with poetic, period-accurate sympathy.
> Ask it about the *Demeter*'s storm in *Dracula*, and it retrieves the actual passage before responding.

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Repository Structure](#repository-structure)
- [The Pipeline](#the-pipeline)
  - [1. Data Collection & Cleaning](#1-data-collection--cleaning)
  - [2. Dialogue Dataset Generation](#2-dialogue-dataset-generation)
  - [3. LoRA Fine-Tuning](#3-lora-fine-tuning)
  - [4. Retrieval-Augmented Generation (RAG)](#4-retrieval-augmented-generation-rag)
- [Dataset Sources](#dataset-sources)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Example Conversation](#example-conversation)
- [Tech Stack](#tech-stack)
- [Roadmap](#roadmap)
- [Limitations](#limitations)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

VictorianGPT is an end-to-end pipeline (data → dataset → fine-tune → retrieval → inference) built and trained entirely on free-tier Google Colab GPUs. It combines two complementary techniques:

| Technique | Purpose |
|---|---|
| **LoRA Supervised Fine-Tuning** | Teaches the base model the *voice, tone, and register* of Victorian English — sentence rhythm, vocabulary, formality, and persona consistency. |
| **RAG (Retrieval-Augmented Generation)** | Grounds responses in real 19th-century text when a user's query touches on plot, characters, or lore from the source novels, and gracefully falls back to a general "Victorian gentleman" persona for everyday conversation. |

The result is a chatbot with a **dynamic dual-mode personality**:
- **Historian Mode** — activated when the retriever finds a strong match in the novel corpus (e.g. questions about *Dracula*, *Jane Eyre*, etc.).
- **Empathetic Companion Mode** — activated when no relevant passage is found (e.g. "I had a rough day at work"), where it responds with Victorian-style emotional support instead of forcing irrelevant lore into the answer.

## How It Works

```
┌─────────────────┐     ┌──────────────────────┐     ┌────────────────────┐     ┌───────────────────────┐
│  1. COLLECTION   │ --> │  2. DATASET BUILD    │ --> │  3. FINE-TUNING     │ --> │  4. RAG + INFERENCE    │
│                  │     │                      │     │                     │     │                        │
│ 40 Gutenberg     │     │ Authentic dialogue   │     │ Qwen2.5-3B-Instruct │     │ ChromaDB vector store  │
│ novels scraped,  │     │ pairs extracted from │     │ + Unsloth LoRA      │     │ + persona-routed       │
│ cleaned & chunked│     │ novels + synthetic   │     │ adapters trained on │     │ generation with        │
│ into sentences   │     │ modern→Victorian     │     │ the master dialogue │     │ Historian/Empathy mode │
│                  │     │ pairs via Qwen2.5-3B │     │ dataset             │     │ switching              │
└─────────────────┘     └──────────────────────┘     └────────────────────┘     └───────────────────────┘
 Notebook 01              Notebook 02                  Notebook 03                Notebook 04
```

## Repository Structure

```
Victorian_GPT/
│
├── README.md                          # You are here
├── requirements.txt                   # Python dependencies for local/Colab use
├── LICENSE                            # License for code (MIT recommended)
├── .gitignore                         # Ignore checkpoints, data dumps, env files
│
├── notebooks/
│   ├── 01_data_collection_and_cleaning.ipynb   # Scrape, clean, sentence-chunk 40 novels
│   ├── 02_dialogue_dataset_generation.ipynb    # Build authentic + synthetic dialogue pairs
│   ├── 03_finetune_qwen_lora.ipynb             # Unsloth LoRA fine-tuning on Qwen2.5-3B
│   └── 04_rag_pipeline_and_inference.ipynb     # ChromaDB RAG + persona-routed chat
│
├── data/
│   ├── raw/                           # (gitignored) raw .txt downloads from Project Gutenberg
│   ├── cleaned/                       # (gitignored) boilerplate-stripped, normalized text
│   ├── chunks_master.parquet          # Sentence-level chunked corpus (sample or full, see note below)
│   ├── chunks_master.csv              # CSV fallback of the same corpus
│   └── dialogues/
│       └── victorian_dataset_master.json   # Final SFT dataset (authentic + synthetic pairs)
│
├── src/                               # (recommended) plain .py scripts refactored from the notebooks
│   ├── scrape.py                      # Notebook 01 as a script
│   ├── clean.py
│   ├── build_dialogue_dataset.py      # Notebook 02 as a script
│   ├── train_lora.py                  # Notebook 03 as a script
│   ├── build_vectorstore.py           # Notebook 04, indexing half
│   └── chat.py                        # Notebook 04, inference/chat half
│
├── models/
│   └── Victorian_Qwen_Final_Adapter/  # (gitignored / Git LFS or Hugging Face Hub) LoRA adapter weights
│
├── vectorstore/
│   └── chroma_db/                     # (gitignored) persisted ChromaDB index
│
├── assets/
│   └── architecture_diagram.png       # Pipeline diagram used in this README
│
└── docs/
    └── PROMPTING.md                   # System prompt design notes / persona rules
```

> **Note on large files:** raw novel text, the full parquet corpus, trained adapter weights, and the ChromaDB index are all large binary/generated artifacts. Don't commit them directly — use a `.gitignore` (see below), and host large weights on the **Hugging Face Hub** or via **Git LFS** instead of the repo itself.

### Suggested `.gitignore`

```gitignore
# Data artifacts
data/raw/
data/cleaned/
vectorstore/chroma_db/
*.zip

# Model artifacts
models/Victorian_Qwen_Final_Adapter/
checkpoints/

# Environments
.env
__pycache__/
*.pyc
.ipynb_checkpoints/

# Colab / Drive
drive/
```

---

## The Pipeline

### 1. Data Collection & Cleaning
**`notebooks/01_data_collection_and_cleaning.ipynb`**

- Downloads **40 curated 19th-century and Gothic novels** directly from [Project Gutenberg](https://www.gutenberg.org/), spanning Dickens, the Brontë sisters, Wilde, Stoker, Shelley, Stevenson, Doyle, Wells, Hardy, Austen, Eliot, Gaskell, Wilkie Collins, and more.
- Strips Project Gutenberg's legal boilerplate (`*** START ... *** END`).
- Normalizes archaic/smart punctuation (curly quotes → straight quotes, em-dashes → hyphens) and flattens whitespace.
- Tokenizes cleaned text into sentences with **NLTK**, then chunks them into overlapping windows (6 sentences per chunk, 2-sentence overlap) to preserve context for downstream embedding.
- Deduplicates and saves the final corpus as both **Parquet** (`chunks_master.parquet`) and **CSV** for portability.
- Archives the full project folder to Google Drive as a single `.zip` for durability across Colab sessions.

### 2. Dialogue Dataset Generation
**`notebooks/02_dialogue_dataset_generation.ipynb`**

Builds the supervised fine-tuning dataset from two complementary sources:

- **Authentic pairs** — extracts every quoted line of dialogue from the cleaned novels via regex, then pairs consecutive quotes as `(input, output)` conversational turns. This teaches the model authentic period cadence, vocabulary, and conversational rhythm straight from the source texts.
- **Synthetic pairs** — generates hundreds of modern, first-person emotional prompts (e.g. *"I feel anxious after an interview"*) across a matrix of emotions × situations × phrasings, then uses **Qwen2.5-3B-Instruct** (zero-shot, via `transformers.pipeline`) with a strict system prompt to translate each one into a natural, non-Shakespearean Victorian-register response. This teaches the model to handle *modern, everyday user input* — which the raw novels don't cover — while staying in character.
- Combines both sources into a single master dataset, tagged by `source` (`authentic_novel` / `synthetic_qwen`), and saves it as `data/dialogues/victorian_dataset_master.json`.

### 3. LoRA Fine-Tuning
**`notebooks/03_finetune_qwen_lora.ipynb`**

- Base model: **Qwen2.5-3B-Instruct**, loaded in 4-bit via **[Unsloth](https://github.com/unslothai/unsloth)** for memory-efficient training on a free Colab T4 GPU.
- Fine-tuning method: **LoRA** (rank 16, α=16, no dropout) applied to all attention and MLP projection matrices (`q/k/v/o_proj`, `gate/up/down_proj`), updating only ~1–2% of total parameters.
- Data is formatted into Qwen's ChatML-style template (`<|im_start|>user ... <|im_end|><|im_start|>assistant ... <|im_end|>`) and trained with Hugging Face's `SFTTrainer` (via `trl`).
- Training config: batch size 2 with 8-step gradient accumulation (effective batch size 16), 1 epoch, learning rate `2e-4`, `adamw_8bit` optimizer, linear LR schedule, mixed precision (bf16/fp16 auto-detected), with periodic checkpointing to Google Drive.
- Final LoRA adapter is saved to `models/Victorian_Qwen_Final_Adapter/`.

### 4. Retrieval-Augmented Generation (RAG)
**`notebooks/04_rag_pipeline_and_inference.ipynb`**

- Embeds every cleaned novel using `sentence-transformers/all-MiniLM-L6-v2` and indexes the chunks into a persistent **ChromaDB** vector store (`vectorstore/chroma_db/`), processed file-by-file to avoid memory overflow on Colab.
- At inference time, the user's query is embedded and matched against the vector store via `similarity_search_with_score`.
- **Confidence-based routing:**
  - If the best match's distance score is **below the threshold** → the retrieved passage is injected as grounding context and the model responds in **Historian Mode**.
  - If the score is **above the threshold** (no relevant lore found) → the model falls back to **Empathetic Companion Mode**, offering period-appropriate emotional support without fabricating irrelevant references to the novels.
- A carefully engineered system prompt enforces the persona: educated, dignified 19th-century London gentleman; strictly modern-Victorian diction (explicitly bans faux-Shakespearean words like *thou/thee/thy/dost*); concise (2–3 sentences); never breaks character or references being an AI.

---

## Dataset Sources

All source texts are in the public domain via **[Project Gutenberg](https://www.gutenberg.org/)**. The corpus spans:

- **Charles Dickens** — *Great Expectations, Oliver Twist, A Tale of Two Cities, David Copperfield, Bleak House, Hard Times*
- **The Brontë Sisters** — *Jane Eyre, Wuthering Heights, The Tenant of Wildfell Hall, Villette*
- **Gothic & Macabre** — *Dracula, Frankenstein, The Picture of Dorian Gray, Dr. Jekyll and Mr. Hyde, The Turn of the Screw, Carmilla*
- **Arthur Conan Doyle** — *The Hound of the Baskervilles, A Study in Scarlet, The Sign of the Four, The Adventures of Sherlock Holmes*
- **H.G. Wells** — *The Time Machine, The War of the Worlds, The Invisible Man, The Island of Dr. Moreau*
- **Thomas Hardy** — *Tess of the d'Urbervilles, Far From the Madding Crowd, Jude the Obscure*
- **Jane Austen** — *Pride and Prejudice, Emma, Sense and Sensibility*
- **George Eliot & Elizabeth Gaskell** — *Middlemarch, Silas Marner, North and South*
- **Wilkie Collins** — *The Woman in White, The Moonstone*
- **Others** — *Alice in Wonderland, The Scarlet Letter, Moby-Dick, Les Misérables*

---

## Getting Started

### Prerequisites
- Python 3.10+
- A CUDA-capable GPU for fine-tuning (the notebooks are built for free-tier Google Colab T4 GPUs)
- A Google Drive account if running the notebooks as-is (used for persistence between Colab sessions)

### Installation

```bash
git clone https://github.com/Hamza-Faarooq/Victorian_GPT.git
cd Victorian_GPT
pip install -r requirements.txt
```

### `requirements.txt` (suggested)

```
requests
beautifulsoup4
pandas
tqdm
nltk
pyarrow
transformers
accelerate
sentencepiece
datasets
trl
peft
bitsandbytes
xformers
unsloth @ git+https://github.com/unslothai/unsloth.git
langchain
langchain-community
langchain-text-splitters
sentence-transformers
chromadb
torch
```

---

## Usage

Run the notebooks in order (each stage depends on the artifacts of the previous one):

1. **`01_data_collection_and_cleaning.ipynb`** — downloads and preprocesses the corpus.
2. **`02_dialogue_dataset_generation.ipynb`** — builds `victorian_dataset_master.json`.
3. **`03_finetune_qwen_lora.ipynb`** — fine-tunes and saves the LoRA adapter.
4. **`04_rag_pipeline_and_inference.ipynb`** — builds the vector store and runs the chatbot.

Each notebook currently persists intermediate artifacts to Google Drive (`/content/drive/MyDrive/VictorianGPT/...`) to survive Colab session resets — update these paths if you run locally or on another platform.

---

## Example Conversation

```
User: After days went well, it seems the day went pretty bad today

[Router] No relevant lore found → Empathetic Companion Mode

VictorianGPT: Ah, my dear friend, even the fairest of days may yield to
an unexpected squall, much as London's own fog descends without warning
upon a clear morning. Take heart — such trials are but fleeting shadows
against the steadier course of one's life.
```

```
User: What happens to the ship Demeter in the storm?

[Router] Relevant passage retrieved from dracula.txt → Historian Mode

VictorianGPT: [Response grounded in the retrieved passage from Dracula]
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data collection | `requests`, `BeautifulSoup`, Project Gutenberg |
| Preprocessing | `re`, `nltk` (sentence tokenization) |
| Dataset generation | `transformers` pipeline, Qwen2.5-3B-Instruct |
| Fine-tuning | `unsloth`, `trl` (`SFTTrainer`), `peft` (LoRA), `bitsandbytes`, `torch` |
| Base model | `Qwen/Qwen2.5-3B-Instruct` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | `ChromaDB` (via `langchain-community`) |
| Orchestration | `langchain`, `langchain-text-splitters` |
| Environment | Google Colab (free T4 GPU), Google Drive persistence |

---

## Roadmap

- [ ] Refactor notebooks into reusable `src/*.py` modules
- [ ] Publish the LoRA adapter to the Hugging Face Hub
- [ ] Add a lightweight Gradio/Streamlit chat UI
- [ ] Add automated eval (perplexity, persona-consistency scoring, retrieval hit-rate)
- [ ] Merge LoRA adapter into base weights for faster inference / quantized export (GGUF)
- [ ] Expand the corpus with additional Victorian-era non-fiction and periodicals

## Limitations

- Trained for one epoch on a relatively small, partly synthetic dataset — responses may occasionally drift from the target persona or produce shorter/generic replies.
- The RAG retrieval threshold is a fixed heuristic and may misroute borderline queries between Historian and Empathy modes.
- Authentic dialogue pairs are extracted by naively pairing *consecutive* quotes in a novel, which does not guarantee true conversational turns (adjacent quotes may belong to different speakers or scenes).
- Built and tested on Google Colab's free tier; GPU/VRAM constraints shaped several design choices (4-bit loading, small batch size, single epoch).

## License

This project's code is intended to be released under the **MIT License** (add a `LICENSE` file to the repo root to make this explicit). All source novels are in the public domain via Project Gutenberg; review [Project Gutenberg's license terms](https://www.gutenberg.org/policy/license.html) before redistributing raw text.

## Acknowledgments

- [Project Gutenberg](https://www.gutenberg.org/) for the public-domain source texts
- [Qwen team](https://github.com/QwenLM/Qwen2.5) for Qwen2.5-3B-Instruct
- [Unsloth](https://github.com/unslothai/unsloth) for efficient LoRA fine-tuning
- [LangChain](https://github.com/langchain-ai/langchain) and [ChromaDB](https://github.com/chroma-core/chroma) for the RAG stack
- [Hugging Face](https://huggingface.co/) for `transformers`, `datasets`, `trl`, `peft`, and `sentence-transformers`
