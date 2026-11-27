# YOHO RAG Retrieval Demo

A small, reproducible project for comparing BM25, Chroma vector retrieval, and
weighted reciprocal rank fusion (RRF) over a sample policy corpus.

## Project layout

```text
.
├── data/
│   ├── policy_chunks.json       # 20 sample chunks
│   ├── test_questions.json      # 12 evaluation questions and expected IDs
│   ├── query_rewrite_questions.json # 10 short queries and 3 rewrite fixtures each
│   ├── hyde_answers.json        # hypothetical answer fixtures for Task 4
│   └── chroma_db/               # local persistent Chroma database (generated)
├── src/
│   ├── retrieval_eval.py        # indexing, retrieval, RRF, Hit@3 evaluation
│   ├── query_rewrite_eval.py    # multi-query RRF and rewrite evaluation
│   └── hyde_eval.py             # raw vs rewrite vs HyDE comparison
├── prompts/
│   ├── query_rewrite.txt        # query rewrite prompt template
│   └── hyde.txt                 # HyDE hypothetical answer prompt
├── reports/
│   ├── query_rewrite_results.md # Task 3 measured results
│   └── hyde_results.md          # Task 4 strategy comparison
├── requirements.txt
└── README.md
```

## Run

```bash
python3 -m pip install -r requirements.txt
python3 src/retrieval_eval.py
python3 src/query_rewrite_eval.py
python3 src/hyde_eval.py
```

The script embeds the chunks with `sentence-transformers/all-MiniLM-L6-v2`,
stores them in the `yoho_policy_chunks` Chroma collection under `data/chroma_db`,
and prints Hit@3 for BM25, vector search, and hybrid vector weights 0.3, 0.5,
and 0.7. The dedicated collection is rebuilt from `data/policy_chunks.json` on
each run, so it stays in sync with the sample corpus. The model must be available
in the local Sentence Transformers cache.

The corpus is synthetic because the workspace did not include source documents.
Replace `data/policy_chunks.json` and update the expected chunk IDs in
`data/test_questions.json` to evaluate a real corpus.

Task 3 uses deterministic, curated rewrite fixtures in
`data/query_rewrite_questions.json`, following `prompts/query_rewrite.txt`.
This allows offline evaluation and reports zero API cost. Its latency measures
the extra local BM25 and Chroma retrieval calls; it excludes LLM generation.

Task 4 compares the raw hybrid query, multi-query rewriting, HyDE, and their
RRF combination. The HyDE prompt is in `prompts/hyde.txt`; static hypothetical
answers are in `data/hyde_answers.json`. See `reports/hyde_results.md` for
Hit@3, measured local latency, cost assumptions, a help/hurt example, and
production guidance. These offline results exclude LLM generation time/cost.
