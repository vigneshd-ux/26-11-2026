# Task 4: HyDE

## Strategy comparison

Measured on the same 10 short queries and 20-chunk sample corpus. The raw
baseline uses BM25 plus Chroma vector rankings fused with RRF. Query rewriting
retrieves the original and three alternatives using both retrievers and fuses
the lists with RRF. HyDE embeds only the hypothetical answer and searches the
Chroma collection. The bonus combines the rewrite rankings and HyDE ranking
with RRF (`k=60`).

| Strategy | Hit@3 | Avg latency/query | Cost |
|---|---:|---:|---:|
| Raw question (BM25 + vector RRF) | 9/10 (0.90) | 0.0079 s | $0.00* |
| Query rewriting (multi-query RRF) | 10/10 (1.00) | 0.0269 s | $0.00* |
| HyDE (hypothetical-answer vector) | 9/10 (0.90) | 0.0068 s | $0.00* |
| Rewrite + HyDE (RRF bonus) | 10/10 (1.00) | 0.0337 s | $0.00* |

Latency is the measured local retrieval and embedding time per query after model
and corpus setup. It excludes LLM hypothetical-answer or rewrite generation.
Cost is zero because these runs used local embeddings and offline fixtures; no
external LLM/API was called.

## Where HyDE helped and hurt

- **Helped — “2FA login?”** The raw hybrid query missed the expected chunk;
  HyDE's hypothetical answer used the fuller concept “second authentication
  factor” and retrieved it in the top 3.
- **Hurt — “privacy officer?”** The offline hypothetical answer drifted toward
  records retention and Legal/Compliance, favoring adjacent chunk C15 over the
  expected privacy-contact chunk C05.

The privacy example is an intentional off-target fixture to show the known risk:
a hypothetical answer can add plausible but unsupported concepts and steer
retrieval away from the user's intent. The fixtures in `data/hyde_answers.json`
are static examples, not outputs from a live model.

## Production guidance

- **Raw query:** Use as the low-latency default for clear, specific questions
  and keep it as the fallback and evaluation baseline.
- **Query rewriting:** Use for short, vague, or abbreviation-heavy questions.
  Keep the number of variants small because each adds retrieval work.
- **HyDE:** Use when a question's wording differs from the corpus vocabulary
  and semantic retrieval needs a richer description. Treat the hypothetical
  answer as a search aid only; verify the final response against retrieved
  source chunks and consider fusing raw-query results to reduce drift.
- **Rewrite + HyDE:** Reserve for difficult queries where recall is important
  enough to justify the additional latency, and deduplicate/rerank the fused
  candidates.

The prompt template is `prompts/hyde.txt`; the reproducible evaluation command
is `python3 src/hyde_eval.py`.
