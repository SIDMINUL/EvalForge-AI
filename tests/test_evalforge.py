from fastapi.testclient import TestClient
from api import app
from evalforge.data import load_dataset,validate_dataset
from evalforge.metrics import evaluate_ranking,mrr_at_k,ndcg_at_k,precision_at_k,recall_at_k
from evalforge.retrieval import BM25Retriever,build_retriever
client=TestClient(app)
def test_dataset():
    d=load_dataset();validate_dataset(d);assert len(d["documents"])>=10 and len(d["queries"])>=10
def test_metrics():
    ids,rel=["a","x","b","y"],["a","b"]
    assert recall_at_k(ids,rel,3)==0.5 and precision_at_k(ids,rel,3)==1/3
    assert mrr_at_k(ids,rel,3)==1.0 and 0<ndcg_at_k(ids,rel,3)<=1
    assert evaluate_ranking(ids,rel,3)["recall@3"]==0.5
def test_bm25():
    docs=load_dataset()["documents"]
    assert BM25Retriever(docs).retrieve("BM25 term frequency inverse document frequency",3)[0]["id"]=="d2"
def test_retrievers():
    docs=load_dataset()["documents"]
    for name in ("bm25","dense","hybrid"):
        rows=build_retriever(name,docs).retrieve("How does recall measure search?",5)
        assert len(rows)==5 and all("id" in r and "score" in r for r in rows)
def test_health():
    r=client.get("/api/health");assert r.status_code==200 and r.json()["service"]=="evalforge-ai"
def test_retrieve_api():
    r=client.post("/api/retrieve",json={"query":"What is BM25?","retriever":"bm25","k":3})
    assert r.status_code==200 and len(r.json()["results"])==3
    assert client.post("/api/retrieve",json={"query":"","retriever":"bm25"}).status_code==422
def test_benchmark_api():
    r=client.post("/api/benchmark",json={"k":3});assert r.status_code==200
    assert {x["retriever"] for x in r.json()["retrievers"]}=={"bm25","dense","hybrid"}
    assert client.get("/api/benchmark/latest").status_code==200

def test_custom_chunking_preserves_headings():
    from api import chunk_document
    text = "INTRODUCTION\n\nFirst section text about freedom.\n\nSECTION 02: HISTORY\n\nHistory contains a different topic."
    chunks = chunk_document("sample.txt", text, chunk_size=100)
    assert len(chunks) >= 2
    assert any("INTRODUCTION" in d["text"] and "First section text" in d["text"] for d in chunks)
    assert not any("First section text" in d["text"] and "History contains" in d["text"] for d in chunks)

def test_question_only_chunks_are_downranked():
    from evalforge.retrieval import BM25Retriever
    docs = [
        {"id": "question", "text": "What does Luffy associate with becoming King of the Pirates?"},
        {"id": "answer", "text": "Luffy associates becoming King of the Pirates with personal freedom, not ruling others."},
    ]
    rows = BM25Retriever(docs).retrieve("What does Luffy associate with becoming King of the Pirates?", 2)
    assert rows[0]["id"] == "answer"
