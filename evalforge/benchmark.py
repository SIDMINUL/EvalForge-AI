"""Reproducible benchmark runner and CLI."""
import argparse,json,statistics
from datetime import datetime,timezone
from pathlib import Path
from evalforge.data import load_dataset
from evalforge.metrics import aggregate_metrics,evaluate_ranking
from evalforge.retrieval import build_retriever,retrieve_timed
RETRIEVERS=("bm25","dense","hybrid")
def percentile(values,p):
    if not values:return 0.0
    values=sorted(values);return values[max(0,min(len(values)-1,int(p/100*len(values)+0.999999)-1))]
def run_benchmark(dataset=None,k=5):
    dataset=dataset or load_dataset();docs,queries=dataset["documents"],dataset["queries"];reports=[]
    for name in RETRIEVERS:
        retriever=build_retriever(name,docs);rows=[];latencies=[]
        for q in queries:
            found,latency=retrieve_timed(retriever,q["query"],k);ids=[r["id"] for r in found]
            rows.append({"query_id":q["id"],"query":q["query"],"relevant_doc_ids":q.get("relevant_doc_ids",[]),"retrieved":found,**evaluate_ranking(ids,q.get("relevant_doc_ids",[]),k),"latency_ms":latency});latencies.append(latency)
        reports.append({"retriever":name,"engine":getattr(retriever,"engine",name),"query_count":len(queries),"metrics":aggregate_metrics(rows,k),"latency_ms":{"mean":statistics.mean(latencies) if latencies else 0,"median":statistics.median(latencies) if latencies else 0,"p95":percentile(latencies,95)},"per_query":rows})
    return {"project":"EvalForge AI","created_at":datetime.now(timezone.utc).isoformat(),"dataset":dataset.get("name","custom"),"document_count":len(docs),"query_count":len(queries),"k":k,"retrievers":reports,"notes":["Results depend on dataset, labels, hardware, and runtime.","TF-IDF fallback is not a neural embedding model.","First run may include model initialization overhead."]}
def main():
    p=argparse.ArgumentParser(description="Benchmark BM25, dense and hybrid retrieval.")
    p.add_argument("--k",type=int,default=5);p.add_argument("--dataset",type=Path,default=None);p.add_argument("--output",type=Path,default=None);a=p.parse_args()
    if a.k<1:p.error("--k must be >= 1")
    r=run_benchmark(load_dataset(a.dataset),a.k)
    print(json.dumps({"project":r["project"],"dataset":r["dataset"],"documents":r["document_count"],"queries":r["query_count"],"k":r["k"],"retrievers":[{"retriever":x["retriever"],"engine":x["engine"],"metrics":x["metrics"],"latency_ms":x["latency_ms"]} for x in r["retrievers"]]},indent=2))
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2),encoding="utf-8");print("Full report written to "+str(a.output))
if __name__=="__main__":main()
