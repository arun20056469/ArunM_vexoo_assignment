# Vexoo Labs – AI Engineer Assignment

## Project Structure

```
├── document_ingestion.py      # Part 1: Sliding window + Knowledge Pyramid + Retrieval
├── retrieval.py               # Part 1: Standalone retrieval interface
├── colab_training_standalone.py  # Part 2: GSM8K fine-tuning with LoRA (Colab)
├── reasoning_adapter.py       # Bonus: Reasoning-aware routing adapter
└── README.md
```

---

## Part 1 — Document Ingestion & Retrieval

### Requirements
```bash
pip install python-dotenv   # no extra deps; uses stdlib only
```

### Run demo
```bash
python document_ingestion.py   # builds pyramid and runs a sample query
python retrieval.py            # standalone retrieval demo
```

### How it works
1. **Sliding window** splits input text into 2500-char chunks with 500-char overlap.
2. **Knowledge Pyramid** builds 4 layers per chunk:
   - `raw` — original text
   - `summary` — first 3 sentences (placeholder; swap in an LLM call)
   - `category` — rule-based keyword classifier → domain label
   - `distilled` — top-8 TF keywords (placeholder for embeddings)
3. **Retrieval** scores chunks using a weighted blend:
   - Fuzzy match on summary (40 %)
   - Keyword overlap on distilled (40 %)
   - Fuzzy match on raw preview (20 %)

To use on your own document:
```python
from document_ingestion import build_pyramid, retrieve
text    = open("my_doc.txt").read()
pyramid = build_pyramid(text)
results = retrieve("your query here", pyramid, top_k=3)
```

---

## Part 2 — GSM8K Fine-tuning (Colab)

### ⚠️ Note
Training was **not executed** due to local compute constraints.
The script is complete and runnable on a Colab T4 or A100.

### Setup (run once in Colab)
```python
!pip install transformers datasets peft accelerate bitsandbytes trl -q
```

### Run
```python
# In a Colab notebook cell:
exec(open("colab_training_standalone.py").read())
```

Or run each section manually — the script is fully commented and modular.

### Key choices
| Decision | Choice | Reason |
|---|---|---|
| Base model | LLaMA-3.2-1B | Fits on Colab T4 in 4-bit |
| Fine-tuning | LoRA (r=16) | <1 % trainable params, fast convergence |
| Quantisation | 4-bit NF4 (bitsandbytes) | Halves VRAM vs fp16 |
| Prompt format | Instruction + chain-of-thought answer | Encourages step-by-step reasoning |
| Evaluation | Exact Match on extracted final number | Standard GSM8K metric |

---

## Bonus — Reasoning Adapter

```bash
python reasoning_adapter.py   # prints routing results for 4 sample queries
```

The adapter detects query domain (math / legal / science / general) via keyword scoring
and dispatches to a specialised module. Confidence score enables escalation to user
clarification when routing is ambiguous.

---

## Environment

- Python 3.10+
- GPU required for Part 2 (Colab T4 recommended)
- No API keys required
