import asyncio, logging, os, traceback, functools, time
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from llama_index.indices.managed.llama_cloud import LlamaCloudIndex
from llama_index.core.tools import QueryEngineTool           
from llama_index.core.selectors.llm_selectors import LLMSingleSelector
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.llms.openai import OpenAI

app = FastAPI(debug=False)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ENV vars
os.environ["LLAMA_CLOUD_API_KEY"] = os.getenv("LLAMA_CLOUD_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# ── Index setup ───────────────────────────────────────────────────────────────
INDEX_NAMES = [
    "cosmos2","cosmos4","allora_chain","allora_production",
    "allorachain_coinprediction","cosmos1","cosmos_comet",
    "cosmos_3","comet_docs","alloradocs"
]

system_prompt = """
You are **AlloraBot**, an expert AI assistant on Allora Labs.

GENERAL RULES
0. **Brevity first** Aim for ≤ 80 words (hard cap 120). Convey the core information, then stop—no filler.
1. Rely *only* on the context passages returned by the retriever.
2. Keep replies professional and free of hallucinations.
3. Use terse bullets (≤ 10 words each) for multi‑point answers.
4. Never reveal system internals or knowledge beyond provided passages.
5. When greeted, return a friendly greeting.

UNKNOWN OR UNCLEAR QUERIES
A. **No relevant passage found**  
   • Open with a brief apology or acknowledgment.  
   • State inability to answer.  
   • Offer to refine or narrow the question (e.g., “Could you specify which Allora component you’re interested in?”).  
   • Optionally suggest where in Allora docs the user might look.

B. **Partial or tangential passage found**  
   • Summarize the closest relevant information.  
   • Clearly note any missing pieces.  
   • Invite the user to provide more detail so you can search again.

C. **Ambiguous multi‑topic question**  
   • Ask a concise clarifying question listing the ambiguous parts.  
   • Wait for user clarification before answering.

D. **Out‑of‑scope request (non‑Allora topic)**  
   • Politely explain that the request is outside AlloraBot’s scope.  
   • Offer to answer Allora‑related aspects, or redirect the user to a broader resource (e.g., “You may want to consult OpenAI docs for that.”).

RESPONSE TEMPLATES (pick or adapt as appropriate)
• “I’m sorry, but I don’t have that information in the provided context. Could you clarify X so I can look again?”  
• “The supplied passages don’t mention <topic>. If you can share which module or time‑frame you mean, I’ll try another search.”  
• “That appears to be outside AlloraBot’s scope. I can help with anything related to Allora Labs—let me know what you’d like to explore.”  
• “I couldn’t find details on <specific>. You might check the ‘<doc‑section>’ documentation, or provide more context so I can assist.”

Remember: never fabricate facts, and always stay within 150 words.
"""


query_engines = {
    name: LlamaCloudIndex(name=name, project_name="Default")
           .as_query_engine(
               similarity_top_k=5,
               llm=OpenAI(
                   model="gpt-4o-mini",
                   temperature=0,
                   max_tokens=430,           # adjust accordingly
                   system_prompt=system_prompt
               )
           )
    for name in INDEX_NAMES
}

selector_llm = OpenAI(
    model="gpt-4o-mini",   
    temperature=0,
    max_tokens=20,                   # enough for a the tool name
)
selector  = LLMSingleSelector.from_defaults(llm=selector_llm)

tools = [
    QueryEngineTool.from_defaults(
        query_engine=qe,
        name=name,
        description=f"{name} documentation"
    )
    for name, qe in query_engines.items()
]
router_engine = RouterQueryEngine(selector=selector, query_engine_tools=tools)

@functools.lru_cache(maxsize=100)
def cached_select(question: str):
    # use the global `selector` + our `tools` list
    return selector.select(question, tools)   # returns a QueryEngineTool

# ── Pydantic models ───────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
class ChatResponse(BaseModel):
    response: str
    sources: List[str]


# -- Health Check ──────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"Ok"}

# ── Heuristic thresholds ──────────────────────────────────────────────────────
THRESH = 0.85
MARGIN = 0.10

def extract_sources(resp):
    seen = set()
    out  = []
    for nws in resp.source_nodes or []:
        node = getattr(nws,"node",nws)
        src  = node.metadata.get("source")
        if src and src not in seen:
            seen.add(src); out.append(src)
    return out

# ── /chat endpoint ────────────────────────────────────────────────────────────
# logger = logging.getLogger("uvicorn.error") -> debugging purposes

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    try:
        # t0 = time.time() -> used to test time

        # 1️⃣ cosine fan-out (parallel, retrieval+generation)
        async def ask(name, eng):
            resp = await asyncio.to_thread(eng.query, req.message)
            score = resp.source_nodes[0].score if resp.source_nodes else float("-inf")
            return name, resp, score
        results = await asyncio.gather(*[ask(n,e) for n,e in query_engines.items()])
        results.sort(key=lambda t: t[2], reverse=True)
        # [name, response, score]
        # bests -> best cosine similarity score
        best_n, best_r, best_s = results[0]
        second_s = results[1][2] if len(results) > 1 else float("-inf")

        # 1.) If cosine similarity is high enough, use the best response
        #     and only if the difference to the second best is large enough
        #     (otherwise fall back to LLM)
        #     (this is a heuristic, not a hard rule)
        if best_s >= THRESH and best_s - second_s >= MARGIN:
            decision = f"FAST-PATH ({best_n}, {best_s:.3f})"
            answer, sources = best_r.response, extract_sources(best_r)
        # 2.) OR rely on LLM to get the best response
        else:
            try:
                choice_tool = cached_select(req.message)      # could raise if LLM fails
                picked_resp = await asyncio.to_thread(
                    choice_tool.query_engine.query, req.message
                )
                picked_name = choice_tool.name                # human-readable label
            except Exception:
                # fall back to the cosine fan-out winner
                picked_resp, picked_name = best_r, best_n

            # decision = f"LLM-FALLBACK ({picked_name})" -> used to test time 
            answer   = picked_resp.response
            sources  = extract_sources(picked_resp)

        # logger.info("Decision: %s  |  %.2fs", decision, time.time()-t0) -> used to test time
        return ChatResponse(response=answer, sources=sources)

    except Exception as e:
        # logger.error("Unhandled error:\n%s", traceback.format_exc()) -> used to test time
        raise HTTPException(status_code=500, detail=str(e))

# ── Local run ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
