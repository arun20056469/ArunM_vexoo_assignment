from document_ingestion import build_pyramid, retrieve

def load_document(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def run_query(query, text, top_k=3):
    pyramid = build_pyramid(text)
    results = retrieve(query, pyramid, top_k=top_k)

    print(f"\nQuery: '{query}'")
    print(f"Retrieved {len(results)} result(s) from {len(pyramid)} chunks\n")
    print("-" * 60)

    for i, r in enumerate(results, 1):
        print(f"Result #{i} | Chunk {r['chunk_id']} | Score: {r['score']} | Category: {r['category']}")
        print(f"Summary: {r['summary'][:120]}")
        print(f"Keywords: {', '.join(r['distilled'])}")
        print(f"Preview: {r['raw_preview'][:200]}")
        print()

if __name__ == "__main__":
    sample = (
        "The Transformer model introduced self-attention to sequence modelling. "
        "BERT pre-trains bidirectional representations from unlabelled text. "
        "GPT uses an autoregressive language model for text generation tasks. "
        "LoRA fine-tuning injects low-rank matrices into transformer weight layers. "
        "Tokenization converts raw text into subword tokens using BPE or WordPiece. "
        "Prompt engineering guides language models with carefully crafted instructions. "
        "RAG combines retrieval systems with generative models for knowledge grounding. "
        "Vector databases store high-dimensional embeddings for fast nearest-neighbour search. "
    ) * 8

    run_query("How does LoRA fine-tune transformer models?", sample, top_k=3)