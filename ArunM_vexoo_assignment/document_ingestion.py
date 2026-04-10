import re
from difflib import SequenceMatcher

def sliding_window(text, window_size=2500, overlap=500):
    chunks = []
    start = 0
    while start < len(text):
        end = start + window_size
        chunks.append(text[start:end])
        start += window_size - overlap
    return chunks

def layer_raw(chunk):
    return chunk.strip()

def layer_summary(chunk, n_sentences=3):
    sentences = re.split(r'(?<=[.!?])\s+', chunk.strip())
    return " ".join(sentences[:n_sentences])

def layer_category(chunk):
    text_lower = chunk.lower()
    rules = {
        "Mathematics": ["equation", "formula", "calculate", "algebra", "geometry", "integral"],
        "Science": ["experiment", "hypothesis", "molecule", "physics", "biology", "chemical"],
        "Legal": ["law", "contract", "clause", "regulation", "jurisdiction", "liability"],
        "Technology": ["software", "algorithm", "neural", "model", "api", "dataset", "training"],
        "History": ["century", "war", "empire", "civilization", "ancient", "revolution"],
        "Healthcare": ["patient", "diagnosis", "treatment", "symptom", "medicine", "clinical"],
    }

    scores = {cat: sum(1 for kw in kws if kw in text_lower) for cat, kws in rules.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "General"

def layer_distilled(chunk, top_n=8):
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to",
        "of", "and", "or", "but", "for", "with", "this", "that", "it", "be",
        "as", "by", "from", "have", "has", "had", "not", "we", "he", "she",
        "they", "you", "i", "can", "will", "do", "does", "its"
    }

    words = re.findall(r'\b[a-zA-Z]{4,}\b', chunk.lower())
    freq = {}

    for w in words:
        if w not in stopwords:
            freq[w] = freq.get(w, 0) + 1

    sorted_kws = sorted(freq, key=freq.get, reverse=True)
    return sorted_kws[:top_n]

def build_pyramid(text):
    chunks = sliding_window(text)
    pyramid = []

    for i, chunk in enumerate(chunks):
        pyramid.append({
            "chunk_id": i,
            "raw": layer_raw(chunk),
            "summary": layer_summary(chunk),
            "category": layer_category(chunk),
            "distilled": layer_distilled(chunk),
        })

    return pyramid

def _fuzzy_score(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def _keyword_score(query, keywords):
    q_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', query.lower()))
    if not q_words:
        return 0.0
    hits = q_words & set(keywords)
    return len(hits) / len(q_words)

def retrieve(query, pyramid, top_k=3):
    results = []

    for entry in pyramid:
        s_summary = _fuzzy_score(query, entry["summary"])
        s_distilled = _keyword_score(query, entry["distilled"])
        s_raw = _fuzzy_score(query, entry["raw"][:500])

        combined = 0.4 * s_summary + 0.4 * s_distilled + 0.2 * s_raw

        results.append({
            "chunk_id": entry["chunk_id"],
            "score": round(combined, 4),
            "category": entry["category"],
            "summary": entry["summary"],
            "distilled": entry["distilled"],
            "raw_preview": entry["raw"][:300] + "...",
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]

if __name__ == "__main__":
    sample_text = (
        "Neural networks are computational models inspired by biological neurons. "
        "Deep learning algorithms learn hierarchical representations of data automatically. "
        "Gradient descent optimizes model weights by minimizing a loss function. "
        "Backpropagation computes gradients efficiently using the chain rule of calculus. "
        "Convolutional neural networks excel at image recognition tasks. "
        "Transformer architectures revolutionized natural language processing with attention mechanisms. "
        "Training large language models requires substantial computational resources and datasets. "
        "Fine-tuning adapts pre-trained models to specific downstream tasks efficiently. "
        "Regularization techniques like dropout prevent overfitting during neural network training. "
        "The bias-variance tradeoff is fundamental to understanding model generalization. "
    ) * 10

    pyramid = build_pyramid(sample_text)

    query = "How does gradient descent optimize neural networks?"
    results = retrieve(query, pyramid, top_k=2)

    for r in results:
        print(f"[Chunk {r['chunk_id']}] Score: {r['score']} | Category: {r['category']}")
        print(f"Summary: {r['summary'][:120]}")
        print(f"Keywords: {r['distilled']}")
        print()