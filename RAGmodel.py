from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import openai

# ── LlamaIndex imports ─────────────────────────────────────────────────────────
# Managed index in LlamaCloud (connects to an existing cloud vector store) :contentReference[oaicite:0]{index=0}
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex
# Query engine interface :contentReference[oaicite:1]{index=1}

# ── FastAPI app and CORS ───────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Environment / Keys ─────────────────────────────────────────────────────────
#  set LLAMA_CLOUD_API_KEY in the env:

llama_key = os.getenv("LLAMA_CLOUD_API_KEY")
if not llama_key:
    raise RuntimeError("Please set LLAMA_CLOUD_API_KEY in your environment")
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise RuntimeError("Please set OPENAI_API_KEY in your environment")



# ── Initialize your managed index + query engine ──────────────────────────────
try:
    # Connect by index name or pipeline ID:
    INDEX_NAME = "allora_production"  # your existing LlamaCloud index name
    index = LlamaCloudIndex(
        name=INDEX_NAME,               
        project_name="Default",        
    )
    # Build a QueryEngine: does retrieval + LLM-synthesis in one call :contentReference[oaicite:2]{index=2}
    query_engine = index.as_query_engine(similarity_top_k=5)
except Exception as e:
    raise RuntimeError(f"Error connecting to LlamaCloud index: {e}")

# ── Pydantic models here
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    sources: List[str]

# Health Check
@app.get("/")
async def root():
    return {"Ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # 1) run RAG query
        resp = query_engine.query(request.message)
        # 2) extract the answer text
        answer = getattr(resp, "response", str(resp))
        # 3) pull out unique source filenames/URLs from retrieved nodes
        sources = []
        for src in getattr(resp, "source_nodes", []):
            md = getattr(src, "node", src).metadata
            # adjust key if you stored it under a different field
            if "source" in md:
                sources.append(md["source"])
        sources = list(set(sources))

        return ChatResponse(response=answer, sources=sources)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Uvicorn runner ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
    
