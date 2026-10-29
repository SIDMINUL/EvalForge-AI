"""BM25, dense and reciprocal-rank-fusion retrievers."""
import math,os,re,time
from collections import Counter
TOKEN_RE=re.compile(r"[a-z0-9]+",re.I)
QUESTION_START=re.compile(r"^(who|what|when|where|why|how|which|does|do|did|is|are|can|could|would|should)\\b",re.I)
def tokenize(text):return TOKEN_RE.findall(text.lower())
def _result(doc,score,name,engine=None):
    out={"id":str(doc["id"]),"text":doc["text"],"score":round(float(score),8),"retriever":name}
    if engine:out["engine"]=engine
    return out
def _question_only_penalty(doc, score):
    """Down-rank question banks and question-only chunks for answer-seeking queries."""
    text=str(doc.get("text","")).strip()
    words=tokenize(text)
    lower=text.lower()
    question_count=text.count("?")
    question_bank_markers=("sample questions", "retrieval test", "discussion prompts",
                           "direct questions include", "thematic questions include",
                           "comparison questions include", "appendix a")
    is_question_bank=question_count >= 3 and (
        any(marker in lower for marker in question_bank_markers) or question_count >= 8
    )
    is_question_only=(len(words) <= 45 and text.endswith("?")
                      and QUESTION_START.match(text) and question_count == 1)
    if is_question_bank:
        return float(score)*0.15
    if is_question_only:
        return float(score)*0.2
    return float(score)
class BM25Retriever:
    name="bm25"
    def __init__(self,documents,k1=1.5,b=0.75):
        self.documents,self.k1,self.b=documents,k1,b
        self.tokens=[tokenize(d["text"]) for d in documents];self.lengths=[len(t) for t in self.tokens]
        self.avgdl=sum(self.lengths)/max(1,len(self.lengths));self.df=Counter()
        for t in self.tokens:self.df.update(set(t))
    def retrieve(self,query,k=5):
        terms,n,scored=tokenize(query),len(self.documents),[]
        for i,doc in enumerate(self.documents):
            tf,score=Counter(self.tokens[i]),0.0
            for term in terms:
                freq=tf.get(term,0)
                if not freq:continue
                df=self.df[term];idf=math.log(1+(n-df+0.5)/(df+0.5))
                denom=freq+self.k1*(1-self.b+self.b*self.lengths[i]/max(1,self.avgdl))
                score+=idf*freq*(self.k1+1)/denom
            score=_question_only_penalty(doc,score)
            scored.append((i,score))
        scored.sort(key=lambda p:(-p[1],str(self.documents[p[0]]["id"])))
        return [_result(self.documents[i],s,self.name) for i,s in scored[:max(0,k)]]
class DenseRetriever:
    name="dense"
    def __init__(self,documents,model_name=None):
        self.documents,self.model,self.vectorizer=documents,None,None;self.engine="tfidf_fallback"
        model_name=model_name or os.getenv("EVALFORGE_EMBEDDING_MODEL","all-MiniLM-L6-v2")
        try:
            from sentence_transformers import SentenceTransformer
            self.model=SentenceTransformer(model_name)
            self.matrix=self.model.encode([d["text"] for d in documents],normalize_embeddings=True,convert_to_numpy=True)
            self.engine="sentence-transformers:"+model_name
        except Exception as exc:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.preprocessing import normalize
            self.vectorizer=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True)
            self.matrix=normalize(self.vectorizer.fit_transform([d["text"] for d in documents]))
            if not isinstance(exc,ImportError):self.engine="tfidf_fallback_after_embedding_error:"+type(exc).__name__
    def retrieve(self,query,k=5):
        if self.model is not None:scores=(self.model.encode([query],normalize_embeddings=True,convert_to_numpy=True)[0]@self.matrix.T).tolist()
        else:
            from sklearn.preprocessing import normalize
            scores=(normalize(self.vectorizer.transform([query]))@self.matrix.T).toarray()[0].tolist()
        scores=[_question_only_penalty(doc,score) for doc,score in zip(self.documents,scores)]
        order=sorted(range(len(self.documents)),key=lambda i:(-float(scores[i]),str(self.documents[i]["id"])))
        return [_result(self.documents[i],scores[i],self.name,self.engine) for i in order[:max(0,k)]]
class HybridRetriever:
    name="hybrid"
    def __init__(self,documents,rrf_k=60):
        self.documents,self.rrf_k=documents,rrf_k
        self.bm25,self.dense=BM25Retriever(documents),DenseRetriever(documents)
        self.engine="rrf(bm25,"+self.dense.engine+")"
    def retrieve(self,query,k=5):
        rankings=[self.bm25.retrieve(query,len(self.documents)),self.dense.retrieve(query,len(self.documents))]
        scores,docs={}, {str(d["id"]):d for d in self.documents}
        for ranking in rankings:
            for rank,item in enumerate(ranking,1):scores[item["id"]]=scores.get(item["id"],0)+1/(self.rrf_k+rank)
        for doc_id in list(scores):
            doc=docs[doc_id]
            scores[doc_id]=_question_only_penalty(doc,scores[doc_id])
        ids=sorted(scores,key=lambda x:(-scores[x],x))
        return [_result(docs[i],scores[i],self.name,self.engine) for i in ids[:max(0,k)]]
def build_retriever(name,documents):
    name=name.lower().strip()
    if name=="bm25":return BM25Retriever(documents)
    if name=="dense":return DenseRetriever(documents)
    if name in {"hybrid","hybrid_rrf"}:return HybridRetriever(documents)
    raise ValueError("Choose bm25, dense, or hybrid.")
def retrieve_timed(retriever,query,k=5):
    start=time.perf_counter();rows=retriever.retrieve(query,k)
    return rows,(time.perf_counter()-start)*1000
