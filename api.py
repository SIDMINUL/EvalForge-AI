"""FastAPI service for EvalForge AI."""
import os
from pathlib import Path
from typing import Literal
from fastapi import FastAPI,HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,Field
from evalforge.benchmark import run_benchmark
from evalforge.data import load_dataset
from evalforge.judge import judge_answer
from evalforge.retrieval import build_retriever,retrieve_timed
BASE=Path(__file__).parent
app=FastAPI(title="EvalForge AI",description="Reproducible evaluation for RAG retrieval pipelines.",version="1.0.0")
app.mount("/static",StaticFiles(directory=str(BASE/"static")),name="static")
_latest=None
class RetrieveRequest(BaseModel):
    query:str=Field(min_length=1,max_length=4000)
    retriever:Literal["bm25","dense","hybrid"]="hybrid"
    k:int=Field(default=5,ge=1,le=20)
class BenchmarkRequest(BaseModel):k:int=Field(default=5,ge=1,le=20)
class JudgeRequest(BaseModel):
    question:str=Field(min_length=1,max_length=4000)
    answer:str=Field(min_length=1,max_length=8000)
    contexts:list[str]=Field(min_length=1,max_length=20)
    expected_answer:str|None=Field(default=None,max_length=4000)
def dataset():
    try:return load_dataset()
    except (OSError,ValueError) as exc:raise HTTPException(status_code=500,detail="Could not load dataset: "+str(exc)) from exc
@app.get("/",include_in_schema=False)
def index():return FileResponse(BASE/"static"/"index.html")
@app.get("/api/health")
def health():
    d=dataset();return {"status":"ok","service":"evalforge-ai","version":app.version,"document_count":len(d["documents"]),"query_count":len(d["queries"]),"embedding_model_configured":bool(os.getenv("EVALFORGE_EMBEDDING_MODEL"))}
@app.get("/api/dataset")
def dataset_summary():
    d=dataset();return {"name":d.get("name","custom"),"description":d.get("description",""),"document_count":len(d["documents"]),"query_count":len(d["queries"]),"documents":d["documents"],"queries":d["queries"]}
@app.post("/api/retrieve")
def retrieve(request:RetrieveRequest):
    d=dataset()
    try:
        engine=build_retriever(request.retriever,d["documents"]);rows,latency=retrieve_timed(engine,request.query,request.k)
        return {"query":request.query,"retriever":request.retriever,"engine":getattr(engine,"engine",request.retriever),"latency_ms":round(latency,3),"results":rows}
    except ValueError as exc:raise HTTPException(status_code=400,detail=str(exc)) from exc
@app.post("/api/benchmark")
def benchmark(request:BenchmarkRequest):
    global _latest
    _latest=run_benchmark(dataset(),request.k);return _latest
@app.get("/api/benchmark/latest")
def latest():
    if _latest is None:raise HTTPException(status_code=404,detail="No benchmark run yet. POST /api/benchmark first.")
    return _latest
@app.post("/api/judge")
def judge(request:JudgeRequest):
    try:return judge_answer(request.question,request.answer,request.contexts,request.expected_answer)
    except RuntimeError as exc:raise HTTPException(status_code=503,detail=str(exc)) from exc
