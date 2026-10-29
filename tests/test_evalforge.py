from fastapi.testclient import TestClient
from api import app
from evalforge.data import load_dataset,validate_dataset
from evalforge.metrics import evaluate_ranking,mrr_at_k,ndcg_at_k,precision_at_k,recall_at_k
from evalforge.retrieval import BM25Retriever,build_retriever
client=TestClient(app)
def test_dataset():
    d=load_dataset();validate_dataset(d);assert len(d["documents"])>=10 and len(d["queries"])>=10
def test_metrics():
    ids, rel = ["a", "x", "b", "y"], ["a", "b"]

    assert recall_at_k(ids, rel, 3) == 1.0
    assert precision_at_k(ids, rel, 3) == 2 / 3
    assert mrr_at_k(ids, rel, 3) == 1.0
    assert 0 < ndcg_at_k(ids, rel, 3) <= 1
    assert evaluate_ranking(ids, rel, 3)["recall@3"] == 1.0
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

def test_question_bank_chunks_are_downranked():
    from evalforge.retrieval import BM25Retriever
    docs = [
        {"id": "questions", "text": "APPENDIX A: SAMPLE QUESTIONS. Direct questions include: What does Luffy want? Why does he value freedom? How does the crew work together? What does Robin study? What is the ship's role?"},
        {"id": "answer", "text": "Luffy associates becoming King of the Pirates with personal freedom, not ruling others. He wants to choose his own path and expects his friends to have the same right."},
    ]
    rows = BM25Retriever(docs).retrieve("What does Luffy associate with becoming King of the Pirates?", 2)
    assert rows[0]["id"] == "answer"

def test_custom_chunking_removes_retrieval_metadata_and_preserves_entity_coverage():
    from api import chunk_document
    text = """SECTION 06: RORONOA ZORO AND DISCIPLINE

Zoro is a swordsman who values discipline, endurance, and loyalty to Luffy.

Key ideas
- Main topic: Zoro and discipline.
- Analytical lens: connect choices to emotional context.
- Retrieval cue: ask a paraphrased question.

SECTION 09: SANJI, FOOD, AND CARE

Sanji is the crew's cook and a capable fighter. He expresses care by feeding his friends.

Key ideas
- Main topic: Sanji and care.
- Analytical lens: connect choices to emotional context.
- Retrieval cue: ask a paraphrased question.
"""
    chunks = chunk_document("characters.txt", text)
    assert chunks
    assert all("Key ideas" not in item["text"] for item in chunks)
    assert all("Retrieval cue" not in item["text"] for item in chunks)

    rows = BM25Retriever(chunks).retrieve("Who is Sanji and Zoro?", 2)
    combined = " ".join(row["text"].lower() for row in rows)
    assert "sanji" in combined
    assert "zoro" in combined


def test_load_dataset_accepts_utf8_bom(tmp_path):
    import json
    from evalforge.data import load_dataset

    payload = {
        "name": "BOM test",
        "documents": [{"id": "d1", "text": "A test passage."}],
        "queries": [{"id": "q1", "query": "test", "relevant_doc_ids": ["d1"]}],
    }
    path = tmp_path / "dataset.json"
    path.write_text("\ufeff" + json.dumps(payload), encoding="utf-8")
    loaded = load_dataset(path)
    assert loaded["name"] == "BOM test"
    assert loaded["queries"][0]["relevant_doc_ids"] == ["d1"]
