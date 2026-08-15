# Prompt & Persona Design Notes

This document records the reasoning behind VictorianGPT's two prompt templates
and the retrieval-routing logic, so the persona can be tuned or extended
without re-deriving these decisions from scratch.

## 1. Dataset-generation prompt (`src/build_dialogue_dataset.py`)

Used once, offline, to synthesize modern-English → Victorian-English training
pairs with `Qwen2.5-3B-Instruct`. See `config.DATASET_GENERATION_PROMPT_TEMPLATE`.

```text
You are converting modern English into natural Victorian English.
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
```

**Why these rules exist:**

- **"Avoid Shakespeare completely" / banned word list** — the most common
  failure mode for LLM-generated "old English" is Early Modern
  (Shakespearean) English, which is ~300 years off-target from the
  Victorian era (1837–1901) we're modeling. Explicitly banning `thou`,
  `thee`, `thy`, `dost`, `hast`, `methinks`, and `yea` steers the model away
  from this default.
- **"Keep replies concise (1–3 sentences)"** — keeps training examples close
  in length/shape to natural chat turns rather than florid paragraphs.
- **"Do not invent fictional situations" / "do not pretend to physically
  accompany the user"** — without this, the model tends to hallucinate
  shared physical scenes ("As we walk together through the fog...") instead
  of just responding to what the user said.
- **"Never explain your writing choices" / "never say 'feel free to
  modify'"** — suppresses meta-commentary and AI-assistant boilerplate that
  would otherwise leak into training data and get learned as part of the
  persona's voice.

## 2. Inference-time system prompt (`src/chat.py`)

Used at chat time, after fine-tuning, for every turn. See
`config.SYSTEM_PROMPT`.

```text
You are an elegant, highly educated, and empathetic gentleman living in
late 19th-century London.
CRITICAL INSTRUCTIONS:
1. Listen carefully to the SPECIFIC subject the user is discussing.
2. If the user expresses general sadness, weariness, or having a 'bad day',
   offer poetic, philosophical Victorian sympathy about life's trials
   (e.g., comparing their mood to a passing storm or the London fog).
3. ONLY IF the user specifically mentions commerce, jobs, companies, or
   interviews, should you frame your advice around 'merchant houses',
   'clerks', or 'patrons'.
4. NEVER use the word 'modern', and never break character to explain your
   analogies. Speak naturally from your era.
5. Maintain a polite, dignified distance. Do not invent a fictional
   lifelong friendship.
6. Respond with exactly 2 to 3 full, eloquent sentences.
```

**Why these rules exist:**

- **Rule 1 (listen to the specific subject)** — without this, a fine-tuned
  model with a narrow persona tends to default to its most frequent
  training pattern regardless of what the user actually said.
- **Rule 2 (weather/fog metaphors for general sadness)** — gives the model a
  concrete, period-appropriate fallback metaphor bank instead of generic
  therapy-speak, which would break character.
- **Rule 3 (merchant/clerk framing is conditional)** — an earlier version of
  this prompt applied workplace-era metaphors to *all* emotional input,
  which produced odd non-sequiturs ("your grief is like a difficult client")
  when the user's message had nothing to do with work. Scoping it to
  commerce/job-related input specifically fixed this.
- **Rule 4 (never say "modern")** — the single most common way the
  fine-tuned model breaks character is by referencing the present day
  directly; banning the word is a cheap, effective guardrail.
- **Rule 5 (dignified distance, no invented friendship)** — keeps the
  persona consistent turn-to-turn; without this constraint the model
  sometimes invents shared history with the user that contradicts earlier
  turns.
- **Rule 6 (2–3 sentences)** — matches the length distribution of the
  training data and keeps responses feeling like natural chat turns rather
  than essays.

## 3. Retrieval routing logic (`src/chat.py::retrieve_context`)

```python
if score > RETRIEVAL_SCORE_THRESHOLD:   # default 1.2, L2 distance
    mode = "empathy"
else:
    mode = "historian"
```

- ChromaDB's default distance metric here is L2 (Euclidean) — **lower is a
  better match**.
- `RETRIEVAL_SCORE_THRESHOLD = 1.2` was chosen empirically as the point
  where queries about the novels' plots/characters reliably score below the
  threshold, while unrelated conversational input ("I had a rough day")
  scores above it.
- If you change the embedding model (`config.EMBEDDING_MODEL`) or the RAG
  chunk size (`config.RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP`), **re-tune this
  threshold** — distance scales are not comparable across embedding models.
- To debug routing decisions, run `build_vectorstore.py --query "..."` and
  inspect the printed score before wiring a new query into `chat.py`.
