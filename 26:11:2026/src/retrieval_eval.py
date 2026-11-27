"""Small reproducible BM25 vs Chroma dense-vector retrieval exercise.

The project uses an explicitly synthetic policy corpus. Its persistent Chroma
database is stored in ./data/chroma_db.
Install requirements.txt before running.
"""
from pathlib import Path
import json
import re
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    # Offline-only compatibility so this sample can be run in restricted
    # environments. Normal runs use rank_bm25, as requested.
    import math
    from collections import Counter

    class BM25Okapi:
        def __init__(self, corpus, k1=1.5, b=0.75):
            self.corpus, self.k1, self.b = corpus, k1, b
            self.lengths = [len(doc) for doc in corpus]
            self.avgdl = sum(self.lengths) / len(self.lengths)
            self.freqs = [Counter(doc) for doc in corpus]
            self.df = Counter(term for freq in self.freqs for term in freq)
        def get_scores(self, query):
            n = len(self.corpus)
            scores = np.zeros(n)
            for i, freq in enumerate(self.freqs):
                for term in query:
                    tf = freq[term]
                    if tf:
                        idf = math.log(1 + (n - self.df[term] + 0.5) / (self.df[term] + 0.5))
                        scores[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avgdl))
            return scores


DEFAULT_VECTOR_WEIGHT = 0.5
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_PATH = PROJECT_ROOT / "data" / "chroma_db"
CHUNKS = json.loads((PROJECT_ROOT / "data" / "policy_chunks.json").read_text())
QUESTIONS = [(q["type"], q["question"], q["expected_chunk_id"])
             for q in json.loads((PROJECT_ROOT / "data" / "test_questions.json").read_text())]
CHROMA_COLLECTION = "yoho_policy_chunks"

def tokenize(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def rrf(rankings, k=60):
    """Fuse ranked ID lists with reciprocal rank fusion (ranks start at 1)."""
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))


def weighted_rrf(rankings, weights, k=60):
    """Weighted RRF; weights correspond positionally to each ranked list."""
    if len(rankings) != len(weights):
        raise ValueError("Provide one weight per ranking")
    scores = {}
    for ranking, weight in zip(rankings, weights):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
    return sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))


def hybrid_ranking(bm25_ranking, vector_ranking, vector_weight, k=60):
    """Fuse BM25 and vector ranks; vector_weight is in [0, 1]."""
    if not 0 <= vector_weight <= 1:
        raise ValueError("vector_weight must be between 0 and 1")
    rankings = [bm25_ranking, vector_ranking]
    if vector_weight == 0.5:
        return rrf(rankings, k=k)
    return weighted_rrf(rankings, [1 - vector_weight, vector_weight], k=k)


def build_retrievers():
    tokenized = [tokenize(c["text"]) for c in CHUNKS]
    bm25 = BM25Okapi(tokenized)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
    corpus_vecs = model.encode([c["text"] for c in CHUNKS], normalize_embeddings=True)

    # Keep this project's vector data in its own local persistent Chroma DB.
    # Rebuild the dedicated collection on each run so it exactly reflects CHUNKS.
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    try:
        client.delete_collection(CHROMA_COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine", "description": "YOHO sample policy chunks"},
    )
    collection.add(
        ids=[c["id"] for c in CHUNKS],
        documents=[c["text"] for c in CHUNKS],
        embeddings=corpus_vecs.tolist(),
        metadatas=[{"source": "synthetic_policy_corpus"} for _ in CHUNKS],
    )
    return bm25, model, collection


def retrieve_rankings(question, bm25, model, collection):
    b_scores = bm25.get_scores(tokenize(question))
    b_ranking = [CHUNKS[i]["id"] for i in np.argsort(b_scores)[::-1]]
    q_vec = model.encode([question], normalize_embeddings=True)
    vector_result = collection.query(
        query_embeddings=q_vec.tolist(),
        n_results=len(CHUNKS),
        include=["distances"],
    )
    return b_ranking, vector_result["ids"][0]


def main():
    bm25, model, collection = build_retrievers()
    per_query = []
    for kind, question, expected in QUESTIONS:
        b_ranking, v_ranking = retrieve_rankings(question, bm25, model, collection)
        per_query.append({"kind": kind, "question": question, "expected": expected,
                          "bm25": b_ranking, "vector": v_ranking})

    methods = {"BM25": [], "Vector": []}
    for weight in (0.3, 0.5, 0.7):
        methods[f"Hybrid (vector={weight:.1f})"] = []
    for row in per_query:
        methods["BM25"].append(row["expected"] in row["bm25"][:3])
        methods["Vector"].append(row["expected"] in row["vector"][:3])
        for vector_weight in (0.3, 0.5, 0.7):
            fused = hybrid_ranking(row["bm25"], row["vector"], vector_weight, k=60)
            methods[f"Hybrid (vector={vector_weight:.1f})"].append(row["expected"] in fused[:3])

    query_count = len(per_query)
    print(f"Hit@3 summary (hits / {query_count})")
    print("Retriever | Hit@3")
    print("---|---:")
    for method, hits in methods.items():
        print(f"{method} | {sum(hits)}/{query_count} ({sum(hits) / len(hits):.2f})")

    print("\nPer-question hits")
    print("Question | BM25 | Vector | Hybrid 0.3 | Hybrid 0.5 | Hybrid 0.7")
    print("---|---|---|---|---|---|")
    for i, row in enumerate(per_query):
        hit = lambda ranking: "Yes" if row["expected"] in ranking[:3] else "No"
        cells = [hit(row["bm25"]), hit(row["vector"])]
        for vector_weight in (0.3, 0.5, 0.7):
            fused = hybrid_ranking(row["bm25"], row["vector"], vector_weight, k=60)
            cells.append(hit(fused))
        print(f"{row['question']} | " + " | ".join(cells))

if __name__ == "__main__":
    main()
