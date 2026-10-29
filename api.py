"""FastAPI service for EvalForge AI."""
import os
import re
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from evalforge.benchmark import run_benchmark
from evalforge.data import load_dataset
from evalforge.judge import judge_answer
from evalforge.retrieval import build_retriever, retrieve_timed

BASE = Path(__file__).parent
app = FastAPI(title="EvalForge AI", description="Reproducible evaluation for RAG retrieval pipelines.", version="1.1.0")
app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")
_latest = None
_custom_docs = []
_custom_filename = None


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    retriever: Literal["bm25", "dense", "hybrid"] = "hybrid"
    k: int = Field(default=5, ge=1, le=20)


class BenchmarkRequest(BaseModel):
    k: int = Field(default=5, ge=1, le=20)


class CustomDocumentRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    content: str = Field(min_length=1, max_length=2_000_000)


class CustomRetrieveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    retriever: Literal["bm25", "dense", "hybrid"] = "hybrid"
    k: int = Field(default=5, ge=1, le=20)


class JudgeRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    answer: str = Field(min_length=1, max_length=8000)
    contexts: list[str] = Field(min_length=1, max_length=20)
    expected_answer: str | None = Field(default=None, max_length=4000)


def dataset():
    try:
        return load_dataset()
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail="Could not load dataset: " + str(exc)) from exc


def chunk_document(filename: str, content: str, chunk_size: int = 1100):
    """Split a user-provided text document into stable, inspectable chunks."""
    text = content.replace("\r\n", "\n").strip()
    if not text:
        raise ValueError("The uploaded document is empty.")
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, current = [], ""
    for paragraph in paragraphs or [text]:
        while len(paragraph) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(paragraph[:chunk_size])
            paragraph = paragraph[chunk_size:]
        if current and len(current) + len(paragraph) + 1 > chunk_size:
            chunks.append(current)
            current = paragraph
        else:
            current = (current + " " + paragraph).strip()
    if current:
        chunks.append(current)
    return [{"id": f"custom-{i + 1}", "text": part, "source": filename} for i, part in enumerate(chunks)]


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(BASE / "static" / "index.html")


@app.get("/api/health")
def health():
    d = dataset()
    return {"status": "ok", "service": "evalforge-ai", "version": app.version, "document_count": len(d["documents"]), "query_count": len(d["queries"]), "embedding_model_configured": bool(os.getenv("EVALFORGE_EMBEDDING_MODEL")), "custom_document_loaded": bool(_custom_docs)}


@app.get("/api/dataset")
def dataset_summary():
    d = dataset()
    return {"name": d.get("name", "custom"), "description": d.get("description", ""), "document_count": len(d["documents"]), "query_count": len(d["queries"]), "documents": d["documents"], "queries": d["queries"]}


@app.post("/api/retrieve")
def retrieve(request: RetrieveRequest):
    d = dataset()
    try:
        engine = build_retriever(request.retriever, d["documents"])
        rows, latency = retrieve_timed(engine, request.query, request.k)
        return {"query": request.query, "retriever": request.retriever, "engine": getattr(engine, "engine", request.retriever), "latency_ms": round(latency, 3), "results": rows}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/experiment/document")
def upload_experiment_document(request: CustomDocumentRequest):
    global _custom_docs, _custom_filename
    suffix = Path(request.filename).suffix.lower()
    if suffix not in {".txt", ".md", ".text"}:
        raise HTTPException(status_code=415, detail="For now, upload a plain-text .txt or Markdown .md file. Export PDF/DOCX text to one of these formats first.")
    try:
        docs = chunk_document(Path(request.filename).name, request.content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _custom_docs = docs
    _custom_filename = Path(request.filename).name
    return {"filename": _custom_filename, "document_count": len(docs), "total_characters": len(request.content), "chunks": [{"id": d["id"], "characters": len(d["text"]), "preview": d["text"][:160]} for d in docs]}


@app.post("/api/experiment/retrieve")
def retrieve_custom_document(request: CustomRetrieveRequest):
    if not _custom_docs:
        raise HTTPException(status_code=400, detail="Upload your own TXT or Markdown document first.")
    try:
        engine = build_retriever(request.retriever, _custom_docs)
        rows, latency = retrieve_timed(engine, request.query, request.k)
        return {"filename": _custom_filename, "query": request.query, "retriever": request.retriever, "engine": getattr(engine, "engine", request.retriever), "latency_ms": round(latency, 3), "results": rows}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/benchmark")
def benchmark(request: BenchmarkRequest):
    global _latest
    _latest = run_benchmark(dataset(), request.k)
    return _latest


@app.get("/api/benchmark/latest")
def latest():
    if _latest is None:
        raise HTTPException(status_code=404, detail="No benchmark run yet. POST /api/benchmark first.")
    return _latest


@app.post("/api/judge")
def judge(request: JudgeRequest):
    try:
        return judge_answer(request.question, request.answer, request.contexts, request.expected_answer)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
