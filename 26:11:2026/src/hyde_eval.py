"""Compare raw-query, multi-query rewrite, and HyDE retrieval strategies."""
import json
import time

from retrieval_eval import CHUNKS, PROJECT_ROOT, build_retrievers, retrieve_rankings, rrf

QUERY_CASES = json.loads(
    (PROJECT_ROOT / "data" / "query_rewrite_questions.json").read_text()
)
HYDE_CASES = json.loads((PROJECT_ROOT / "data" / "hyde_answers.json").read_text())


def vector_ranking(search_text, model, collection, chunk_count):
    """Embed the supplied text and return its Chroma ranking."""
    query_embedding = model.encode([search_text], normalize_embeddings=True)
    result = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=chunk_count,
        include=["distances"],
    )
    return result["ids"][0]


def main():
    bm25, model, collection = build_retrievers()
    chunk_count = len(CHUNKS)
    results = {"Raw question (BM25 + vector RRF)": [],
               "Query rewriting (multi-query RRF)": [],
               "HyDE (hypothetical-answer vector)": [],
               "Rewrite + HyDE (RRF bonus)": []}
    elapsed = {name: 0.0 for name in results}

    for question_case, hyde_case in zip(QUERY_CASES, HYDE_CASES):
        question = question_case["question"]
        expected = question_case["expected_chunk_id"]
        if question != hyde_case["question"] or expected != hyde_case["expected_chunk_id"]:
            raise ValueError("HyDE fixtures must align with the query rewrite evaluation cases")

        start = time.perf_counter()
        raw_bm25, raw_vector = retrieve_rankings(question, bm25, model, collection)
        raw_fused = rrf([raw_bm25, raw_vector], k=60)
        elapsed["Raw question (BM25 + vector RRF)"] += time.perf_counter() - start
        results["Raw question (BM25 + vector RRF)"].append(expected in raw_fused[:3])

        start = time.perf_counter()
        rewrite_rankings = []
        for rewritten_query in [question, *question_case["rewrites"]]:
            b_rank, v_rank = retrieve_rankings(rewritten_query, bm25, model, collection)
            rewrite_rankings.extend([b_rank, v_rank])
        rewrite_fused = rrf(rewrite_rankings, k=60)
        rewrite_duration = time.perf_counter() - start
        elapsed["Query rewriting (multi-query RRF)"] += rewrite_duration
        results["Query rewriting (multi-query RRF)"].append(expected in rewrite_fused[:3])

        start = time.perf_counter()
        hyde_vector = vector_ranking(
            hyde_case["hypothetical_answer"], model, collection, chunk_count
        )
        hyde_duration = time.perf_counter() - start
        elapsed["HyDE (hypothetical-answer vector)"] += hyde_duration
        results["HyDE (hypothetical-answer vector)"].append(expected in hyde_vector[:3])

        start = time.perf_counter()
        combined_fused = rrf([*rewrite_rankings, hyde_vector], k=60)
        fusion_duration = time.perf_counter() - start
        elapsed["Rewrite + HyDE (RRF bonus)"] += (
            rewrite_duration + hyde_duration + fusion_duration
        )
        results["Rewrite + HyDE (RRF bonus)"].append(expected in combined_fused[:3])

    n = len(QUERY_CASES)
    print("Strategy | Hit@3 | Avg latency/query | Cost")
    print("---|---:|---:|---:|")
    for name, hits in results.items():
        latency = elapsed[name] / n
        print(f"{name} | {sum(hits)}/{n} ({sum(hits)/n:.2f}) | {latency:.4f} s | $0.00*")

    helps, hurts = [], []
    raw_hits = results["Raw question (BM25 + vector RRF)"]
    hyde_hits = results["HyDE (hypothetical-answer vector)"]
    for case, raw_hit, hyde_hit in zip(QUERY_CASES, raw_hits, hyde_hits):
        if hyde_hit and not raw_hit:
            helps.append(case["question"])
        if raw_hit and not hyde_hit:
            hurts.append(case["question"])
    print("\nHyDE helps vs raw:", ", ".join(helps) if helps else "none on this set")
    print("HyDE hurts vs raw:", ", ".join(hurts) if hurts else "none on this set")
    print("*Offline fixture run: no LLM/API calls. Latency excludes answer generation.")


if __name__ == "__main__":
    main()
