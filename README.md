# EvalForge AI

**A RAG evaluation workbench for measuring retrieval quality, inspecting results, and judging generated answers.**

EvalForge AI helps developers compare retrieval strategies on a labelled dataset and experiment with their own documents. It brings retrieval, ranking metrics, latency measurement, and an optional hosted LLM judge into one web dashboard.

**Live demo:** https://autostream-ai-agent-g5hi.onrender.com

## What it does

- **Compare retrieval methods:** BM25, dense retrieval, and hybrid retrieval with Reciprocal Rank Fusion (RRF).
- **Measure retrieval quality:** Recall@K, Precision@K, Mean Reciprocal Rank (MRR@K), and Normalized Discounted Cumulative Gain (NDCG@K).
- **Inspect performance:** Review retrieved documents and measure retrieval latency, including median and P95.
- **Experiment with your own content:** Upload TXT or Markdown documents, create searchable chunks, and query them using the available retrievers.
- **Evaluate generated answers:** Use the optional Groq-hosted LLM judge to score groundedness, relevance, and correctness.
- **Export and reproduce:** Run the benchmark from the CLI, export JSON reports, and use the included tests.

## Tech stack

- **Backend:** Python, FastAPI
- **Retrieval:** BM25, Sentence Transformers (optional), TF-IDF fallback, Reciprocal Rank Fusion
- **Evaluation:** Recall@K, Precision@K, MRR@K, NDCG@K, latency
- **LLM judge:** Groq API
- **Frontend:** HTML, CSS, JavaScript
- **Testing:** pytest
- **Deployment:** Render

## Getting started

Python 3.10+ is recommended.

```bash
git clone https://github.com/SIDMINUL/AutoStream-AI-Agent.git
cd AutoStream-AI-Agent

python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell
# .venv\\Scripts\\Activate.ps1

pip install -r requirements.txt
uvicorn api:app --reload
```

Open the dashboard at http://127.0.0.1:8000. Interactive API documentation is available at http://127.0.0.1:8000/docs.

### Optional neural dense retrieval

Install Sentence Transformers to use the neural embedding backend:

```bash
pip install sentence-transformers
```

The first run may download the configured embedding model. If Sentence Transformers is unavailable, EvalForge AI uses its TF-IDF cosine-similarity fallback and identifies that backend in the results.

## Run a benchmark

```bash
python -m evalforge.benchmark --k 5 --output reports/benchmark.json
```

The benchmark uses the labelled sample corpus in `evalforge/sample_data.json`. Each query contains relevant document IDs, allowing the evaluation metrics to be calculated against known relevance labels.

## Run an experiment on your own document

1. Open the dashboard's **Your document** section.
2. Upload a TXT or Markdown file (maximum 2 MB).
3. Enter a question and select BM25, dense, or hybrid retrieval.
4. Inspect the retrieved chunks and latency.

The uploaded document is held in server memory and is replaced when another document is uploaded. Restarting the service clears it. PDF and DOCX uploads are not currently supported.

Custom-document mode is intended for retrieval exploration. Ranking metrics such as Recall@K and NDCG@K require labelled queries and relevant document IDs; the application does not invent relevance labels for uploaded documents.

## Optional LLM-as-a-Judge

EvalForge AI can use Groq-hosted inference to evaluate a generated answer against its question and retrieved context.

Set these environment variables on the server:

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes, for judging | Your Groq API key |
| `GROQ_MODEL` | No | Model ID enabled for your Groq account; defaults to `openai/gpt-oss-120b` |
| `EVALFORGE_DATASET` | No | Path to a custom dataset JSON |
| `EVALFORGE_EMBEDDING_MODEL` | No | Sentence Transformers model name |

The judge returns scores for **groundedness**, **relevance**, and **correctness**, along with a rationale. These are model-generated assessments and should be reviewed by a human. Ollama is not required on the deployed Render service.

## API overview

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Service and dataset status |
| GET | `/api/dataset` | Sample dataset and counts |
| POST | `/api/retrieve` | Retrieve from the sample corpus |
| POST | `/api/experiment/document` | Upload a TXT/Markdown document |
| POST | `/api/experiment/retrieve` | Retrieve from the uploaded document |
| POST | `/api/benchmark` | Run the labelled benchmark |
| GET | `/api/benchmark/latest` | Retrieve the latest in-memory report |
| POST | `/api/judge` | Run the optional Groq answer judge |

Example retrieval request:

```json
{
  "query": "How do I evaluate RAG retrieval quality?",
  "retriever": "hybrid",
  "k": 5
}
```

## Evaluation metrics

- **Recall@K:** Fraction of all relevant documents retrieved in the top K.
- **Precision@K:** Fraction of the top K retrieved documents that are relevant.
- **MRR@K:** Reciprocal rank of the first relevant result, averaged across queries.
- **NDCG@K:** Ranking metric that rewards relevant documents appearing higher in the result list.
- **Latency:** Time taken by the retrieval operation; results depend on hardware, caching, and model warm-up.

## Exploratory benchmark snapshot

An exploratory local run on 29 October 2026 used an expanded corpus of 43 documents and 74 queries. Relevance labels for 10 queries were manually broadened after reviewing overlapping passages.

| Retriever | Recall@5 | Precision@5 | MRR@5 | NDCG@5 | Median latency (ms) | P95 latency (ms) |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.8649 | 0.1973 | 0.6475 | 0.6972 | 0.438 | 0.723 |
| Dense (all-MiniLM-L6-v2) | 0.9347 | 0.2135 | 0.7523 | 0.7960 | 13.922 | 21.385 |
| Hybrid RRF | 0.9414 | 0.2135 | 0.7959 | 0.8211 | 12.604 | 15.575 |

**Interpretation:** Hybrid RRF achieved the strongest aggregate ranking metrics in this particular run. Treat these values as an exploratory snapshot, not a production-grade or independently validated result. The corpus and queries were synthetic, 10 query labels were revised manually, and latency is environment-dependent. The expanded dataset and report are not included in the starter repository, so these exact figures are not reproducible from a fresh checkout.

## Tests

```bash
pip install pytest httpx
pytest -q
```

GitHub Actions runs the test suite on pushes and pull requests.

## Current limitations

- The included sample dataset is small and illustrative.
- TF-IDF fallback is not equivalent to a neural embedding model.
- Uploaded documents and the latest benchmark report are stored in memory and are cleared when the service restarts.
- The optional LLM judge requires a valid Groq API key and access to the selected model.
- Production use would benefit from persistent experiment storage, dataset versioning, access control, and token/cost tracking.

## Maintainer

[SIDMINUL](https://github.com/SIDMINUL)
