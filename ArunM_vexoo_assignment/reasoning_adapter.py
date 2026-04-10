import re

def math_reasoner(query):
    return (
        f"[MATH MODULE]\n"
        f"Query     : {query}\n"
        f"Approach  : Step-by-step numeric decomposition\n"
        f"Steps     : [1] Parse numbers/operators → [2] Apply BODMAS → [3] Verify result\n"
        f"Simulated Answer: <run symbolic solver here>"
    )

def legal_reasoner(query):
    return (
        f"[LEGAL MODULE]\n"
        f"Query     : {query}\n"
        f"Approach  : Retrieve relevant clauses → match jurisdiction → cite statute\n"
        f"Simulated Answer: <run legal retrieval + citation engine here>"
    )

def science_reasoner(query):
    return (
        f"[SCIENCE MODULE]\n"
        f"Query     : {query}\n"
        f"Approach  : Identify claim → retrieve evidence → evaluate hypothesis\n"
        f"Simulated Answer: <run scientific reasoning chain here>"
    )

def general_reasoner(query):
    return (
        f"[GENERAL MODULE]\n"
        f"Query     : {query}\n"
        f"Approach  : Semantic similarity search over knowledge pyramid\n"
        f"Simulated Answer: <run RAG retrieval here>"
    )

DOMAIN_KEYWORDS = {
    "math": [
        "calculate", "compute", "equation", "solve", "integral", "derivative",
        "probability", "algebra", "geometry", "how many", "sum", "total",
        "percentage", "formula", "arithmetic", "matrix", "vector",
    ],
    "legal": [
        "law", "legal", "contract", "clause", "regulation", "statute", "liable",
        "jurisdiction", "rights", "court", "penalty", "compliance", "gdpr",
        "intellectual property", "patent", "sue", "damages",
    ],
    "science": [
        "molecule", "atom", "physics", "chemical", "biology", "hypothesis",
        "experiment", "energy", "force", "cell", "dna", "evolution",
        "quantum", "relativity", "species", "compound",
    ],
}

class ReasoningAdapter:

    def detect_type(self, query):
        query_lower = query.lower()
        scores = {}

        for domain, keywords in DOMAIN_KEYWORDS.items():
            scores[domain] = sum(1 for kw in keywords if kw in query_lower)

        best = max(scores, key=scores.get)
        confidence = scores[best]

        if confidence == 0:
            return "general", 0.0

        return best, float(confidence)

    def route(self, query):
        domain, confidence = self.detect_type(query)

        dispatch = {
            "math": math_reasoner,
            "legal": legal_reasoner,
            "science": science_reasoner,
            "general": general_reasoner,
        }

        handler = dispatch.get(domain, general_reasoner)
        response = handler(query)

        return {
            "query": query,
            "domain": domain,
            "confidence": confidence,
            "response": response,
        }

if __name__ == "__main__":
    adapter = ReasoningAdapter()

    queries = [
        "What is the derivative of x squared plus 3x?",
        "Is a verbal contract legally binding in India?",
        "How does DNA replication work in eukaryotic cells?",
        "Tell me about attention mechanisms in transformers.",
    ]

    for q in queries:
        result = adapter.route(q)
        print(f"\nQuery: {result['query']}")
        print(f"Domain: {result['domain']} (confidence: {result['confidence']})")
        print(result["response"])