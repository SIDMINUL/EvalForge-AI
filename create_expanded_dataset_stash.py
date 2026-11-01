import json
from pathlib import Path

from evalforge.data import validate_dataset

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "evalforge" / "sample_data.json"
OUTPUT = ROOT / "evalforge" / "expanded_data.json"

with SOURCE.open(encoding="utf-8") as f:
    original = json.load(f)

# Preserve the original documents and add focused, answer-bearing passages.
topics = [
    ("retrieval", "Retrieval selects passages from a document collection using a query. A retrieval system should return relevant evidence before answer generation.", [
        ("What does retrieval do before answer generation?", "direct"),
        ("How does a RAG system find evidence for an answer?", "paraphrase"),
    ]),
    ("generation", "Grounded generation uses retrieved passages as evidence. The language model should answer from that evidence and avoid unsupported claims.", [
        ("How should a language model use retrieved evidence?", "direct"),
        ("Why should generated answers avoid unsupported claims?", "scenario"),
    ]),
    ("bm25", "BM25 ranks documents using term frequency, inverse document frequency, and document-length normalization. Its lexical scoring is effective when query terms overlap with relevant text.", [
        ("Which factors determine a BM25 score?", "direct"),
        ("Why can BM25 work well for keyword search?", "paraphrase"),
    ]),
    ("dense", "Dense retrieval encodes queries and passages as numerical vectors. Vector similarity can match semantically related text even when exact keywords differ.", [
        ("How can vector retrieval match different wording?", "paraphrase"),
        ("What representations are compared in dense retrieval?", "direct"),
    ]),
    ("hybrid", "Hybrid retrieval combines complementary ranking signals, such as lexical BM25 scores and dense semantic rankings. Reciprocal Rank Fusion can combine ranked lists without requiring comparable raw scores.", [
        ("Why combine lexical and semantic retrieval?", "comparison"),
        ("How does Reciprocal Rank Fusion merge rankings?", "direct"),
    ]),
    ("rrf", "Reciprocal Rank Fusion assigns each document a contribution based on one divided by a constant plus its rank. Contributions from multiple rankings are summed to produce a fused ordering.", [
        ("What score contribution does RRF assign to a ranked document?", "direct"),
        ("Why can RRF combine systems with different score scales?", "paraphrase"),
    ]),
    ("recall", "Recall at k is the number of relevant documents retrieved in the first k results divided by the total number of known relevant documents for the query.", [
        ("How is Recall@5 calculated?", "direct"),
        ("Which metric measures the fraction of relevant documents found?", "paraphrase"),
    ]),
    ("precision", "Precision at k is the number of relevant results among the first k results divided by k. It reflects how much of the returned result list is relevant.", [
        ("How do you calculate Precision@5?", "direct"),
        ("Which metric penalizes irrelevant items in the top results?", "paraphrase"),
    ]),
    ("mrr", "Mean Reciprocal Rank averages the reciprocal of the rank of the first relevant result across queries. A relevant result at rank one contributes one, while lower ranks contribute less.", [
        ("What does MRR measure?", "direct"),
        ("Why does finding the first relevant passage at rank one improve MRR?", "paraphrase"),
    ]),
    ("ndcg", "Normalized Discounted Cumulative Gain rewards relevant documents near the top of a ranked list. It normalizes discounted gain against an ideal ranking.", [
        ("What ranking behavior does NDCG reward?", "direct"),
        ("Why does a relevant document at rank one receive more gain than one at rank five?", "paraphrase"),
    ]),
    ("labels", "A retrieval benchmark needs query-level relevance judgments that identify which documents genuinely answer each query. Incorrect or incomplete labels can distort all ranking metrics.", [
        ("Why must relevance labels be audited?", "scenario"),
        ("How can incorrect ground truth affect retrieval evaluation?", "paraphrase"),
    ]),
    ("dataset", "A reliable evaluation dataset should cover representative topics and query styles, document its source, and keep development examples separate from held-out test examples.", [
        ("What makes a retrieval evaluation dataset reliable?", "direct"),
        ("Why should test queries be held out during development?", "scenario"),
    ]),
    ("latency", "Retrieval latency measures the time required to return results. A fair report should distinguish warm-up from steady-state runs and include median and tail latency.", [
        ("Which latency statistics should a benchmark report?", "direct"),
        ("Why separate model warm-up from steady-state timing?", "scenario"),
    ]),
    ("judge", "An LLM-as-a-judge uses a language model and a scoring rubric to assess generated answers for relevance, correctness, and support from the supplied context.", [
        ("What does an LLM judge evaluate?", "direct"),
        ("How can a language model score answer quality?", "paraphrase"),
    ]),
    ("citations", "Citation accuracy measures whether cited sources support the claims associated with them. Groundedness measures whether answer claims are supported by the provided evidence.", [
        ("What is the difference between citation accuracy and groundedness?", "comparison"),
        ("How do you check whether a citation supports a claim?", "direct"),
    ]),
    ("chunking", "Document chunking divides long documents into smaller passages for indexing and retrieval. Chunk boundaries should preserve useful context and avoid splitting related content unnecessarily.", [
        ("Why does chunking affect retrieval quality?", "scenario"),
        ("How should a document be divided for retrieval?", "paraphrase"),
    ]),
    ("embeddings", "Text embeddings map text into vector representations. Similarity between embeddings can help identify passages with related meanings rather than relying only on shared words.", [
        ("What is a text embedding?", "direct"),
        ("How can embeddings help semantic search?", "paraphrase"),
    ]),
    ("reranking", "A reranker scores an initial candidate set using a more detailed relevance model and reorders the candidates. It can improve top-result ordering at the cost of extra computation.", [
        ("What does a reranker do after initial retrieval?", "direct"),
        ("What is the trade-off of adding a reranking stage?", "comparison"),
    ]),
    ("hallucination", "A hallucination is a generated claim that is not supported by reliable evidence. Retrieval can provide context, but the system must still check whether each answer claim follows from that context.", [
        ("What is a hallucination in a RAG answer?", "direct"),
        ("Does retrieving context automatically guarantee grounded answers?", "scenario"),
    ]),
    ("testing", "Retrieval tests should include expected relevant document IDs and assert that ranking behavior remains correct after code changes. Regression tests help detect accidental quality degradation.", [
        ("How can regression tests protect retrieval quality?", "direct"),
        ("What should a retrieval test assert?", "paraphrase"),
    ]),
    ("reproducibility", "Reproducible benchmarks record the dataset version, relevance labels, configuration, software environment, and measurement procedure so results can be compared fairly.", [
        ("What information makes benchmark results reproducible?", "direct"),
        ("Why record the retrieval configuration and dataset version?", "scenario"),
    ]),
    ("topk", "The value k determines how many top-ranked results are evaluated. Metrics at different k values answer different questions about the breadth and quality of retrieved results.", [
        ("What does k mean in Recall@k?", "direct"),
        ("Why can Recall@5 differ from Recall@10?", "comparison"),
    ]),
    ("similarity", "Cosine similarity compares the angle between nonzero vectors. When embeddings are normalized, their dot product corresponds to cosine similarity.", [
        ("What does cosine similarity compare?", "direct"),
        ("What is the relationship between normalized vectors and cosine similarity?", "paraphrase"),
    ]),
    ("filtering", "Metadata filtering restricts the candidate document set using attributes such as category, date, or access permissions before or during retrieval.", [
        ("What is metadata filtering in search?", "direct"),
        ("How can document attributes constrain retrieval candidates?", "paraphrase"),
    ]),
    ("failure", "A retrieval failure occurs when relevant evidence is missed or ranked too low. Error analysis should inspect the query, relevant labels, retrieved passages, and competing documents.", [
        ("How should a retrieval failure be investigated?", "scenario"),
        ("What evidence helps explain a bad ranking?", "paraphrase"),
    ]),
    ("tradeoffs", "Lexical search is often efficient and interpretable for exact terminology, while dense search can handle semantic variation. The best choice depends on data, queries, quality requirements, and latency limits.", [
        ("When might dense retrieval help more than lexical search?", "comparison"),
        ("How should a team choose between BM25 and dense retrieval?", "scenario"),
    ]),
    ("monitoring", "Production retrieval monitoring can track latency, errors, query distributions, result quality, and changes in document coverage. Offline benchmarks should be supplemented with real-world feedback.", [
        ("Which signals are useful for monitoring production retrieval?", "direct"),
        ("Why are offline benchmarks not enough on their own?", "scenario"),
    ]),
    ("security", "A retrieval system should enforce document access controls so that search results do not expose information the requesting user is not authorized to read.", [
        ("Why should retrieval enforce access controls?", "direct"),
        ("What is a risk of ignoring permissions during search?", "scenario"),
    ]),
    ("deduplication", "Duplicate or near-duplicate passages can crowd the top results and reduce result diversity. Deduplication can improve the usefulness of a result list when repeated content is common.", [
        ("How can duplicate passages hurt search results?", "scenario"),
        ("Why might a retrieval system deduplicate results?", "paraphrase"),
    ]),
    ("evaluationleakage", "Evaluation leakage occurs when test queries or their answers influence model selection or tuning. A held-out test set helps estimate performance on unseen queries.", [
        ("What is evaluation leakage?", "direct"),
        ("Why should test answers not guide repeated tuning?", "scenario"),
    ]),
    ("thresholds", "A similarity threshold can reject weak matches, but thresholds must be calibrated on representative examples. An overly strict threshold may discard useful evidence.", [
        ("What is a risk of setting a retrieval threshold too high?", "scenario"),
        ("How should a similarity threshold be selected?", "paraphrase"),
    ]),
]

documents = list(original["documents"])
queries = []
seen_doc_ids = {d["id"] for d in documents}

# Build 30 new topical passages and two labelled query variants per passage.
# The original starter records remain unchanged.
for index, (topic, text, query_specs) in enumerate(topics, start=13):
    doc_id = f"d{index}"
    if doc_id in seen_doc_ids:
        raise ValueError(f"Duplicate generated document ID: {doc_id}")
    documents.append({"id": doc_id, "text": text})
    for variant, (question, style) in enumerate(query_specs, start=1):
        queries.append({
            "id": f"expanded_{topic}_{variant}",
            "query": question,
            "relevant_doc_ids": [doc_id],
            "query_style": style,
        })

# Keep the original starter queries for a comparable baseline.
# Add the new, explicitly labelled queries.
expanded = {
    "name": "EvalForge Expanded Retrieval Benchmark",
    "description": (
        "Expanded synthetic benchmark for development and error analysis. "
        "Labels are constructed from the passage associated with each query; "
        "independent review is still recommended before making general claims."
    ),
    "documents": documents,
    "queries": original["queries"] + queries,
}

validate_dataset(expanded)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT.open("w", encoding="utf-8") as f:
    json.dump(expanded, f, indent=2, ensure_ascii=False)
    f.write("\n")

print(f"Created: {OUTPUT}")
print(f"Documents: {len(expanded['documents'])}")
print(f"Queries: {len(expanded['queries'])}")
print("Validation: passed")
