# Task 3: Query Rewriting

## Retrieval results

Evaluation on 10 short or vague questions using the 20-chunk synthetic policy
corpus. Without rewriting, the original query uses BM25 plus Chroma vector
retrieval fused with RRF. With rewriting, the original and three alternative
queries are each retrieved by BM25 and Chroma, then all eight ranked lists are
merged with RRF (`k=60`).

| Query | Expected chunk | Hit@3 without rewrite | Hit@3 with rewrite |
|---|---|---:|---:|
| wfh rules? | C17 | Yes | Yes |
| notice period | C18 | Yes | Yes |
| sick days? | C19 | Yes | Yes |
| parental leave | C20 | Yes | Yes |
| meal cap? | C02 | Yes | Yes |
| 401k match? | C04 | Yes | Yes |
| privacy officer? | C05 | Yes | Yes |
| report a breach | C06 | Yes | Yes |
| 2FA login? | C01 | No | Yes |
| receipt for claims? | C16 | Yes | Yes |

**Hit@3 without rewrite: 9/10 (0.90)**  
**Hit@3 with rewrite: 10/10 (1.00)**

## Latency and cost

- Average extra local retrieval latency: **+0.040 seconds per query** on the recorded run.
- Rewrite API cost: **$0.00**; the run used curated offline rewrite fixtures and made no LLM/API calls.
- The latency figure includes the additional BM25 and Chroma retrievals. It excludes LLM generation time, so it is not a production end-to-end latency estimate.

## Prompt and evaluation notes

The prompt template is in `prompts/query_rewrite.txt`. The three alternatives
for each question are stored in `data/query_rewrite_questions.json` to keep the
evaluation repeatable without network access or API credentials. They are
curated prompt-style outputs, not live LLM generations.

Run with `python3 src/query_rewrite_eval.py`.
