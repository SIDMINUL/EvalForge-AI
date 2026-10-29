"""Ranking metrics with explicit definitions."""
import math
def recall_at_k(retrieved_ids,relevant_ids,k=5):
    rel=set(map(str,relevant_ids));return len(set(map(str,list(retrieved_ids)[:k]))&rel)/len(rel) if rel else 0.0
def precision_at_k(retrieved_ids,relevant_ids,k=5):
    if k<=0:return 0.0
    rel=set(map(str,relevant_ids));return sum(x in rel for x in list(map(str,retrieved_ids))[:k])/k
def mrr_at_k(retrieved_ids,relevant_ids,k=5):
    rel=set(map(str,relevant_ids))
    for rank,doc in enumerate(list(map(str,retrieved_ids))[:k],1):
        if doc in rel:return 1/rank
    return 0.0
def ndcg_at_k(retrieved_ids,relevant_ids,k=5):
    rel=set(map(str,relevant_ids));rows=list(map(str,retrieved_ids))[:k]
    dcg=sum(1/math.log2(rank+1) for rank,doc in enumerate(rows,1) if doc in rel)
    ideal=min(len(rel),k);idcg=sum(1/math.log2(rank+1) for rank in range(1,ideal+1))
    return dcg/idcg if idcg else 0.0
def evaluate_ranking(retrieved_ids,relevant_ids,k=5):
    return {"recall@"+str(k):recall_at_k(retrieved_ids,relevant_ids,k),"precision@"+str(k):precision_at_k(retrieved_ids,relevant_ids,k),"mrr@"+str(k):mrr_at_k(retrieved_ids,relevant_ids,k),"ndcg@"+str(k):ndcg_at_k(retrieved_ids,relevant_ids,k)}
def aggregate_metrics(rows,k=5):
    keys=["recall@"+str(k),"precision@"+str(k),"mrr@"+str(k),"ndcg@"+str(k)]
    return {key:sum(r[key] for r in rows)/len(rows) for key in keys} if rows else {key:0.0 for key in keys}
