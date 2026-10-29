const $=s=>document.querySelector(s);let report=null;
async function req(url,opts={}) {
  let r;
  try {
    r = await fetch(url, opts);
  } catch (err) {
    throw Error("Could not reach the API. Check your connection and try again.");
  }
  const contentType = r.headers.get("content-type") || "";
  const body = await r.text();
  let data;
  if (contentType.includes("application/json")) {
    try { data = JSON.parse(body); }
    catch { throw Error("The API returned invalid JSON (HTTP " + r.status + "). Please retry; if this continues, check the deployment logs."); }
  } else {
    const preview = body.replace(/\s+/g, " ").slice(0, 180);
    throw Error("Expected JSON from " + url + " but received " + (contentType || "an unknown content type") + " (HTTP " + r.status + "). " + (preview ? "Response: " + preview : "The response was empty."));
  }
  if (!r.ok) throw Error(data.detail || data.message || ("Request failed (HTTP " + r.status + ")"));
  return data;
}
function fmt(v,n=3){return Number.isFinite(Number(v))?Number(v).toFixed(n):"—"}
function esc(v){return String(v).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]))}
function msg(sel,text,kind=""){const e=$(sel);e.textContent=text;e.className="message"+(kind?" "+kind:"")}
async function health(){try{const d=await req("/api/health");$("#health-text").textContent="API online";$("#hero-docs").textContent=d.document_count+" docs";$("#hero-queries").textContent=d.query_count+" queries";$("#metric-docs").textContent=d.document_count;$("#metric-queries").textContent=d.query_count}catch(e){$("#health-text").textContent="API unavailable";msg("#benchmark-message",e.message,"error")}}
function renderQueries(){if(!report)return;const filter=$("#query-filter").value,rows=report.retrievers.filter(r=>filter==="all"||r.retriever===filter).flatMap(r=>r.per_query.map(q=>({...q,method:r.retriever,k:report.k})));$("#query-rows").innerHTML=rows.map(q=>{const ids=q.retrieved.map(d=>d.id),miss=q.relevant_doc_ids.filter(id=>!ids.includes(id));return "<tr><td>"+esc(q.query)+"</td><td><span class='tag'>"+esc(q.method)+"</span></td><td>"+fmt(q["recall@"+q.k])+"</td><td>"+fmt(q["mrr@"+q.k])+"</td><td>"+q.relevant_doc_ids.map(esc).join(", ")+(miss.length?"<small class='miss'>Missed: "+miss.map(esc).join(", ")+"</small>":"")+"</td><td>"+q.retrieved.slice(0,3).map(d=>"<span class='result-id'>"+esc(d.id)+"</span>").join(" ")+"</td></tr>"}).join("")}
function render(d){report=d;$("#metric-cutoff").textContent="@"+d.k;$("#benchmark-rows").innerHTML=d.retrievers.map(r=>{const m=r.metrics;return "<tr><td><strong>"+esc(r.retriever.toUpperCase())+"</strong><small class='subcell'>"+esc(r.engine)+"</small></td><td>"+fmt(m["recall@"+d.k])+"</td><td>"+fmt(m["precision@"+d.k])+"</td><td>"+fmt(m["mrr@"+d.k])+"</td><td>"+fmt(m["ndcg@"+d.k])+"</td><td>"+fmt(r.latency_ms.median,2)+"</td><td>"+fmt(r.latency_ms.p95,2)+"</td></tr>"}).join("");renderQueries();msg("#benchmark-message","Completed "+d.query_count+" queries across "+d.retrievers.length+" retrievers. Engine details are shown.","success")}
async function run(){$("#run-benchmark").disabled=$("#run-hero").disabled=true;msg("#benchmark-message","Running retrievers. First dense run may initialize the embedding engine…");try{render(await req("/api/benchmark",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({k:Number($("#k-value").value)})}))}catch(e){msg("#benchmark-message",e.message,"error")}finally{$("#run-benchmark").disabled=$("#run-hero").disabled=false}}
$("#run-benchmark").addEventListener("click",run);$("#run-hero").addEventListener("click",()=>{document.querySelector("#benchmark").scrollIntoView({behavior:"smooth"});run()});$("#query-filter").addEventListener("change",renderQueries);
$("#retrieve-form").addEventListener("submit",async e=>{e.preventDefault();const b=e.submitter;b.disabled=true;msg("#retrieval-message","Retrieving passages…");$("#retrieval-results").innerHTML="";try{const d=await req("/api/retrieve",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({query:$("#query-input").value,retriever:$("#retriever-select").value,k:Number($("#retrieve-k").value)})});msg("#retrieval-message",d.results.length+" results · "+fmt(d.latency_ms,2)+" ms · engine: "+d.engine,"success");$("#retrieval-results").innerHTML=d.results.map((r,i)=>"<article class='result-card'><div class='result-meta'><span>#"+(i+1)+" · "+esc(r.id)+"</span><b>score "+fmt(r.score,5)+"</b></div><p>"+esc(r.text)+"</p></article>").join("")}catch(err){msg("#retrieval-message",err.message,"error")}finally{b.disabled=false}});
$("#judge-form").addEventListener("submit",async e=>{e.preventDefault();const b=e.submitter;b.disabled=true;$("#judge-result").hidden=true;msg("#judge-message","Asking Groq to evaluate the answer…");try{const contexts=$("#judge-context").value.split("\n").map(s=>s.trim()).filter(Boolean),d=await req("/api/judge",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({question:$("#judge-question").value,answer:$("#judge-answer").value,contexts,expected_answer:$("#judge-expected").value||null})});$("#judge-result").textContent=JSON.stringify(d,null,2);$("#judge-result").hidden=false;msg("#judge-message","Judge completed with "+d.model+" via Groq. Review scores before relying on them.","success")}catch(err){msg("#judge-message",err.message,"error")}finally{b.disabled=false}});
health();

$("#custom-upload-form").addEventListener("submit", async e => {
  e.preventDefault();
  const b = e.submitter, file = $("#custom-file").files[0];
  if (!file) return;
  if (file.size > 2 * 1024 * 1024) { msg("#custom-upload-message", "File is larger than 2 MB. Choose a smaller TXT or Markdown file.", "error"); return; }
  b.disabled = true;
  $("#custom-chunks").innerHTML = "";
  $("#custom-retrieval-results").innerHTML = "";
  msg("#custom-upload-message", "Reading and chunking your document…");
  try {
    const content = await file.text();
    const d = await req("/api/experiment/document", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({filename:file.name, content})});
    msg("#custom-upload-message", "Loaded " + d.filename + " · " + d.document_count + " searchable chunks · " + d.total_characters + " characters.", "success");
    $("#custom-chunks").innerHTML = "<article class='result-card'><div class='result-meta'><span>Document preview</span><b>" + d.document_count + " chunks</b></div><p>" + d.chunks.map(c => "<span class='result-id'>" + esc(c.id) + "</span> " + esc(c.preview)).join("<br><br>") + "</p></article>";
    msg("#custom-retrieval-message", "Document ready. Ask a question to test retrieval.");
  } catch (err) { msg("#custom-upload-message", err.message, "error"); }
  finally { b.disabled = false; }
});
$("#custom-retrieve-form").addEventListener("submit", async e => {
  e.preventDefault();
  const b = e.submitter;
  b.disabled = true;
  $("#custom-retrieval-results").innerHTML = "";
  msg("#custom-retrieval-message", "Searching your document chunks…");
  try {
    const d = await req("/api/experiment/retrieve", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({query:$("#custom-query").value, retriever:$("#custom-retriever").value, k:Number($("#custom-k").value)})});
    msg("#custom-retrieval-message", d.results.length + " results from " + d.filename + " · " + fmt(d.latency_ms,2) + " ms · engine: " + d.engine, "success");
    $("#custom-retrieval-results").innerHTML = d.results.map((r,i) => "<article class='result-card'><div class='result-meta'><span>#" + (i+1) + " · " + esc(r.id) + "</span><b>score " + fmt(r.score,5) + "</b></div><p>" + esc(r.text) + "</p></article>").join("");
  } catch (err) { msg("#custom-retrieval-message", err.message, "error"); }
  finally { b.disabled = false; }
});
