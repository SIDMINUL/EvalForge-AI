# EvalForge AI

A reproducible evaluation workbench for retrieval-augmented generation (RAG). Compare BM25, dense, and hybrid retrieval; inspect ranking quality and latency; optionally score generated answers with a local LLM judge.

## Features
- BM25 lexical retrieval implemented in-project.
- Dense retrieval with optional Sentence Transformers and a TF-IDF cosine fallback.
- Hybrid retrieval using Reciprocal Rank Fusion (RRF).
- Recall@k, Precision@k, MRR@k, NDCG@k, and latency metrics.
- Labelled sample corpus, per-query inspection, JSON report export.
- FastAPI service, responsive dashboard, CLI benchmark, tests and CI.
- Optional Ollama LLM-as-a-judge for groundedness, relevance, and correctness.

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
- POST /api/retrieve — retrieve with BM25, dense, or hybrid
- POST /api/benchmark — run labelled benchmark
- GET /api/benchmark/latest — latest in-memory report
- POST /api/judge — optional Ollama evaluation

Example retrieval JSON: {"query":"How do I evaluate RAG retrieval quality?","retriever":"hybrid","k":5}

## Dataset and metrics
evalforge/sample_data.json contains stable document IDs and relevant_doc_ids for each query. The sample is illustrative, not production-grade. Review labels and use held-out queries before making performance claims. Recall@k measures relevant documents found in top k; Precision@k is relevant results divided by k; MRR@k rewards the first relevant result's rank; NDCG@k rewards relevant results near the top. Latency depends on hardware, cache, and model warm-up.

## Optional LLM judge
Start Ollama and pull a model such as llama3.2. Configure OLLAMA_BASE_URL (default http://localhost:11434) and OLLAMA_MODEL (default llama3.2), then call POST /api/judge. Scores are model estimates and need human review.

## Configuration
EVALFORGE_DATASET optionally points to a custom dataset JSON. EVALFORGE_EMBEDDING_MODEL selects the embedding model. OLLAMA_BASE_URL and OLLAMA_MODEL configure the local judge.

## Development
    pip install pytest httpx
    pytest -q

GitHub Actions runs tests on pushes and pull requests.

## Limitations
Sample data is small; TF-IDF fallback is not a neural embedding model; latest benchmark history is in memory unless exported; the judge requires Ollama. Production use should add dataset versioning, persistent experiment storage, access control, and token/cost accounting.

Maintainer: https://github.com/SIDMINUL
