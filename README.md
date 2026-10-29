# EvalForge AI

A reproducible evaluation workbench for retrieval-augmented generation (RAG). Compare BM25, dense, and hybrid retrieval; inspect ranking quality and latency; optionally score generated answers with a local LLM judge.

## Features
- BM25 lexical retrieval implemented in-project.
- Dense retrieval with optional Sentence Transformers and a TF-IDF cosine fallback.
- Hybrid retrieval using Reciprocal Rank Fusion (RRF).
- Recall@k, Precision@k, MRR@k, NDCG@k, and latency metrics.
- Labelled sample corpus, per-query inspection, JSON report export.
- Upload your own TXT/Markdown document and run a retrieval experiment against automatically created chunks.
- FastAPI service, responsive dashboard, CLI benchmark, tests and CI.
- Optional Groq-hosted LLM-as-a-judge for groundedness, relevance, and correctness.

## Quick start
Python 3.10+ recommended.

    git clone https://github.com/SIDMINUL/AutoStream-AI-Agent.git
    cd AutoStream-AI-Agent
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    uvicorn api:app --reload

Windows activation: .venv\\Scripts\\activate. Dashboard: http://127.0.0.1:8000. API docs: http://127.0.0.1:8000/docs.

Run a benchmark and export JSON:

    python -m evalforge.benchmark --k 5 --output reports/benchmark.json

For neural embeddings, install Sentence Transformers with pip install sentence-transformers. The first run may download all-MiniLM-L6-v2. Without it, dense retrieval uses TF-IDF and reports that fallback engine explicitly.

## API
- GET /api/health — service and dataset status
- GET /api/dataset — dataset examples and counts
- POST /api/retrieve — retrieve with BM25, dense, or hybrid on the sample corpus
- POST /api/experiment/document — upload TXT/Markdown content and split it into searchable chunks
- POST /api/experiment/retrieve — query the currently uploaded custom document
- POST /api/benchmark — run labelled benchmark
- GET /api/benchmark/latest — latest in-memory report
- POST /api/judge — optional Groq LLM evaluation

Example retrieval JSON: {"query":"How do I evaluate RAG retrieval quality?","retriever":"hybrid","k":5}

## Dataset and metrics
evalforge/sample_data.json contains stable document IDs and relevant_doc_ids for each query. The sample is illustrative, not production-grade. Review labels and use held-out queries before making performance claims. Recall@k measures relevant documents found in top k; Precision@k is relevant results divided by k; MRR@k rewards the first relevant result's rank; NDCG@k rewards relevant results near the top. Latency depends on hardware, cache, and model warm-up.

## Exploratory expanded benchmark results

A local exploratory run on 29 October 2026 used an expanded corpus of 43 documents and 74 queries, with Recall@5, Precision@5, MRR@5, NDCG@5, and retrieval latency. The relevance labels for 10 queries were manually broadened after inspecting overlapping passages.

| Retriever | Recall@5 | Precision@5 | MRR@5 | NDCG@5 | Median latency (ms) | P95 latency (ms) |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.8649 | 0.1973 | 0.6475 | 0.6972 | 0.438 | 0.723 |
| Dense (all-MiniLM-L6-v2) | 0.9347 | 0.2135 | 0.7523 | 0.7960 | 13.922 | 21.385 |
| Hybrid RRF | 0.9414 | 0.2135 | 0.7959 | 0.8211 | 12.604 | 15.575 |

**Interpretation:** Hybrid RRF had the strongest aggregate ranking metrics in this run. The benchmark is exploratory, not a production-grade or independently validated evaluation: the corpus and queries are synthetic, only 10 query labels were revised, and latency is hardware/runtime dependent. These figures are recorded for transparency; the expanded dataset and report are not part of the checked-in starter corpus, so they cannot be reproduced from a fresh checkout yet. Do not describe label revisions as an algorithmic improvement.

## Run an experiment on your own document
Use the **Your document** section in the dashboard to upload a `.txt` or `.md` file (maximum 2 MB), then ask questions against its chunks using BM25, dense, or hybrid retrieval. The chunker preserves detected headings and avoids merging chunks across section boundaries. Short question-only chunks are down-ranked for answer-style retrieval so a repeated question in an appendix is less likely to outrank an explanatory passage. The upload is held in the running server's memory and is replaced when another document is uploaded; restarting the service clears it. PDF/DOCX upload is not supported yet, so export or copy the document text to TXT/Markdown first. This mode shows retrieved chunks and latency. Meaningful Recall@k, Precision@k, MRR@k, and NDCG@k benchmarking requires labelled queries and relevant document IDs; the app does not fabricate labels for personal documents.

## Optional LLM judge
Configure GROQ_API_KEY in the server environment. GROQ_MODEL is optional and defaults to llama-3.3-70b-versatile. The judge calls Groq's hosted chat-completions API, so Ollama does not need to run on the Render service. Then call POST /api/judge. Scores are model estimates and need human review.

## Configuration
EVALFORGE_DATASET optionally points to a custom dataset JSON. EVALFORGE_EMBEDDING_MODEL selects the embedding model. GROQ_API_KEY authenticates the hosted judge; GROQ_MODEL selects the model.

## Development
    pip install pytest httpx
    pytest -q

GitHub Actions runs tests on pushes and pull requests.

## Limitations
Sample data is small; TF-IDF fallback is not a neural embedding model; latest benchmark history is in memory unless exported; the optional judge requires a valid Groq API key and available model. Production use should add dataset versioning, persistent experiment storage, access control, and token/cost accounting.

Maintainer: https://github.com/SIDMINUL
