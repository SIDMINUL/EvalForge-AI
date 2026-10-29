"""Load and validate retrieval evaluation datasets."""
import json,os
from pathlib import Path

DEFAULT_DATASET=Path(__file__).with_name("sample_data.json")

def load_dataset(path=None):
    source=Path(path or os.getenv("EVALFORGE_DATASET",DEFAULT_DATASET))
    # utf-8-sig accepts ordinary UTF-8 and files saved with a UTF-8 BOM
    # (for example, by Windows PowerShell's Set-Content).
    with source.open(encoding="utf-8-sig") as f:
        data=json.load(f)
    validate_dataset(data)
    return data

def validate_dataset(data):
    if not isinstance(data,dict):raise ValueError("Dataset must be a JSON object.")
    docs,queries=data.get("documents"),data.get("queries")
    if not isinstance(docs,list) or not docs:raise ValueError("Dataset needs documents.")
    if not isinstance(queries,list) or not queries:raise ValueError("Dataset needs queries.")
    ids=set()
    for d in docs:
        if not isinstance(d,dict) or not d.get("id") or not d.get("text"):raise ValueError("Each document needs id and text.")
        if str(d["id"]) in ids:raise ValueError("Duplicate document id: "+str(d["id"]))
        ids.add(str(d["id"]))
    qids=set()
    for q in queries:
        if not isinstance(q,dict) or not q.get("id") or not q.get("query"):raise ValueError("Each query needs id and query.")
        if str(q["id"]) in qids:raise ValueError("Duplicate query id: "+str(q["id"]))
        qids.add(str(q["id"]))
        unknown=set(map(str,q.get("relevant_doc_ids",[])))-ids
        if unknown:raise ValueError("Unknown relevant IDs: "+", ".join(sorted(unknown)))
