"""Evaluate multi-query rewriting with offline rewrite fixtures and RRF."""
import json
import time
from pathlib import Path

from retrieval_eval import CHUNKS, PROJECT_ROOT, build_retrievers, retrieve_rankings, rrf

QUESTIONS_PATH = PROJECT_ROOT / "data" / "query_rewrite_questions.json"
QUESTIONS = json.loads(QUESTIONS_PATH.read_text())


def main():
    bm25, model, collection = build_retrievers()
    without_rewrite = []
    with_rewrite = []
    baseline_seconds = 0.0
    rewritten_seconds = 0.0

    for item in QUESTIONS:
        expected = item["expected_chunk_id"]

        start = time.perf_counter()
        bm25_ranking, vector_ranking = retrieve_rankings(
            item["question"], bm25, model, collection
        )
        base_ranking = rrf([bm25_ranking, vector_ranking], k=60)
        baseline_seconds += time.perf_counter() - start
        without_rewrite.append(expected in base_ranking[:3])

        # Rewrites are deterministic fixtures prepared from prompts/query_rewrite.txt.
        # This keeps the exercise runnable without an external LLM/API key.
        start = time.perf_counter()
        query_rankings = []
        for query in [item["question"], *item["rewrites"]]:
            b_ranking, v_ranking = retrieve_rankings(query, bm25, model, collection)
            query_rankings.extend([b_ranking, v_ranking])
        rewritten_ranking = rrf(query_rankings, k=60)
        rewritten_seconds += time.perf_counter() - start
        with_rewrite.append(expected in rewritten_ranking[:3])

    count = len(QUESTIONS)
    base_hits = sum(without_rewrite)
    rewrite_hits = sum(with_rewrite)
    extra_seconds = rewritten_seconds / count - baseline_seconds / count

    print("Query | Expected | Hit@3 without rewrite | Hit@3 with rewrite")
    print("---|---|---|---|")
    for item, base_hit, rewrite_hit in zip(QUESTIONS, without_rewrite, with_rewrite):
        print(
            f"{item['question']} | {item['expected_chunk_id']} | "
            f"{'Yes' if base_hit else 'No'} | {'Yes' if rewrite_hit else 'No'}"
        )

    print("\nSummary")
    print(f"Hit@3 without rewrite: {base_hits}/{count} ({base_hits / count:.2f})")
    print(f"Hit@3 with rewrite: {rewrite_hits}/{count} ({rewrite_hits / count:.2f})")
    print(f"Average extra retrieval latency: +{extra_seconds:.3f} s/query")
    print("Rewrite API cost: $0.00 (offline fixtures; no LLM/API call made)")
    print("Latency excludes LLM rewrite generation; it measures extra local retrieval only.")
    print(f"Corpus chunks: {len(CHUNKS)}")


if __name__ == "__main__":
    main()
