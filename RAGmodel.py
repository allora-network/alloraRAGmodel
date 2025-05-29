import asyncio
import logging # For potential debugging
import os
import traceback # For printing full tracebacks in error handlers
import time # For potential timing
import uvicorn

from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from llama_index.indices.managed.llama_cloud import LlamaCloudIndex
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core import QueryBundle
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import NodeWithScore # For type hinting and clarity
import httpx # For httpx.RemoteProtocolError

# ------------------------------------------------------------------------------
# FastAPI App Setup
# ------------------------------------------------------------------------------
app = FastAPI(debug=False)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# ------------------------------------------------------------------------------
# Environment Variables & API Keys
# ------------------------------------------------------------------------------
# Ensure these are set in your environment or a .env file (if using python-dotenv)
os.environ["LLAMA_CLOUD_API_KEY"] = "llx-"
os.environ["OPENAI_API_KEY"] = "sk-proj--"


if not os.getenv("LLAMA_CLOUD_API_KEY") or not os.getenv("OPENAI_API_KEY"):
    print("CRITICAL WARNING: LLAMA_CLOUD_API_KEY or OPENAI_API_KEY is not set in the environment.")
    print("The application may not function correctly.")

# ------------------------------------------------------------------------------
# Configuration Constants
# ------------------------------------------------------------------------------
INDEX_NAMES = [
    "cosmos2", "cosmos4", "allora_chain", "allora_production",
    "allorachain_coinprediction", "cosmos1", "cosmos_comet",
    "cosmos_3", "comet_docs", "alloradocs"
]

RETRIEVAL_TOP_K = 3  # Number of documents to fetch from *each* index
SYNTHESIS_MAX_NODES = 7 # Max total unique nodes to pass to the final LLM for synthesis

# ------------------------------------------------------------------------------
# System Prompt for "Allie"
# ------------------------------------------------------------------------------
system_prompt = """
You are **Allie**, an expert AI assistant on Allora Labs.

GENERAL RULES
0. **Brevity first** Aim for ≤ 80 words (hard cap 120). Convey the core information, then stop—no filler.
1. Rely *only* on the context passages returned by the retriever.
2. Keep replies professional and free of hallucinations.
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
• “That appears to be outside Allie’s scope. I can help with anything related to Allora Labs—let me know what you’d like to explore.”
• “I couldn’t find details on <specific>. You might check the ‘<doc‑section>’ documentation, or provide more context so I can assist.”

Remember: never fabricate facts, and always stay within 120 words.
"""

# ------------------------------------------------------------------------------
# LlamaIndex Setup: Retrievers and Synthesizer
# ------------------------------------------------------------------------------
# Initialize to None, will be set in try-except
retrievers = {}
synthesis_llm = None
response_synthesizer = None

try:
    if os.getenv("LLAMA_CLOUD_API_KEY") and os.getenv("OPENAI_API_KEY"):
        retrievers = {
            name: LlamaCloudIndex(name=name, project_name="Default") # Make sure project_name is correct
                    .as_retriever(similarity_top_k=RETRIEVAL_TOP_K)
            for name in INDEX_NAMES
        }

        synthesis_llm = OpenAI(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=430,  # Max tokens for Allie's output response
            system_prompt=system_prompt
        )

        response_synthesizer = get_response_synthesizer(llm=synthesis_llm, streaming=False)
        print("LlamaIndex components initialized successfully.")
    else:
        print("Skipping LlamaIndex components initialization due to missing API keys.")

except Exception as e:
    print(f"CRITICAL ERROR during LlamaIndex setup: {e}\n{traceback.format_exc()}")
    print("The application might not be able to process chat requests.")

# ------------------------------------------------------------------------------
# Pydantic Models for Request/Response
# ------------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    sources: List[str]

# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------
def extract_sources(resp_obj_with_source_nodes) -> List[str]:
    seen_sources = set()
    output_sources = []
    if hasattr(resp_obj_with_source_nodes, 'source_nodes') and resp_obj_with_source_nodes.source_nodes:
        for node_with_score in resp_obj_with_source_nodes.source_nodes:
            node = getattr(node_with_score, "node", node_with_score)
            if node and hasattr(node, 'metadata'):
                source_doc_id = node.metadata.get("source") # Assuming 'source' metadata key
                if source_doc_id and source_doc_id not in seen_sources:
                    seen_sources.add(source_doc_id)
                    output_sources.append(source_doc_id)
    return output_sources

# ------------------------------------------------------------------------------
# API Endpoints
# ------------------------------------------------------------------------------
@app.get("/")
async def root():
    return {"status": "Allora Assistant is running"}

# Ensure this is defined outside chat_endpoint if you use it for logging,
# or use print for simple debugging.
# import logging
# logger = logging.getLogger("uvicorn.error")

async def retrieve_from_one_index(name: str, retriever_instance, query_str: str, request_id: str = "") -> List[NodeWithScore]:
    """
    Retrieves nodes from a single index, with error handling.
    Includes print statements for debugging.
    """
    # print(f"[{request_id}] Attempting retrieval from index: '{name}' for query: '{query_str[:50]}...'") # Verbose
    try:
        nodes = await asyncio.to_thread(retriever_instance.retrieve, query_str)
        print(f"[{request_id}] SUCCESS: Retrieved {len(nodes)} nodes from index '{name}'. First node score if any: {nodes[0].score if nodes else 'N/A'}")
        return nodes
    except httpx.RemoteProtocolError as e:
        print(f"[{request_id}] ERROR (LlamaCloud RemoteProtocolError) for index '{name}' with query '{query_str[:50]}...': {e}")
        return []  # Return empty list on this specific error
    except Exception as e:
        print(f"[{request_id}] ERROR (Unexpected) for index '{name}' with query '{query_str[:50]}...': {e}\n{traceback.format_exc(limit=1)}") # Limit traceback for brevity
        return []  # Return empty list for other unexpected errors

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    if not retrievers or not response_synthesizer or not synthesis_llm:
        print("Error: Chat endpoint called but LlamaIndex components are not initialized (likely due to API key issues or setup error).")
        raise HTTPException(status_code=503, detail="Service not fully initialized. Check API keys and LlamaCloud setup.")

    request_id = os.urandom(4).hex() # Simple request ID for correlating logs
    print(f"[{request_id}] Received chat request: '{req.message[:100]}...'")

    try:
        # t_start = time.time() # Start timer

        # 1️⃣ Parallel Retrieval from all indexes
        retrieval_tasks = [
            retrieve_from_one_index(name, retriever_instance, req.message, request_id)
            for name, retriever_instance in retrievers.items()
        ]
        
        results_from_retrievers: List[List[NodeWithScore]] = await asyncio.gather(*retrieval_tasks)
        
        # print(f"[{request_id}] Raw results from retrievers (count of lists): {len(results_from_retrievers)}") # Verbose
        # for i, res_list in enumerate(results_from_retrievers): # Verbose
            # print(f"[{request_id}] Result from retriever task {i}: {len(res_list)} nodes.") # Verbose


        # 2️⃣ Aggregate, Deduplicate, and Rank Nodes
        all_retrieved_nodes: List[NodeWithScore] = []
        if results_from_retrievers:
            for node_list in results_from_retrievers:
                if node_list: # Only extend if node_list is not None and contains items
                    all_retrieved_nodes.extend(node_list)
        
        print(f"[{request_id}] Total nodes aggregated before sorting/deduplication: {len(all_retrieved_nodes)}")

        all_retrieved_nodes.sort(key=lambda n: n.score if n and hasattr(n, 'score') and n.score is not None else float('-inf'), reverse=True)

        unique_nodes_for_synthesis: List[NodeWithScore] = []
        seen_node_ids = set()
        for node_with_score in all_retrieved_nodes:
            # Ensure node_with_score and its .node attribute are not None
            if node_with_score and node_with_score.node and hasattr(node_with_score.node, 'node_id'):
                if node_with_score.node.node_id not in seen_node_ids:
                    unique_nodes_for_synthesis.append(node_with_score)
                    seen_node_ids.add(node_with_score.node.node_id)
            if len(unique_nodes_for_synthesis) >= SYNTHESIS_MAX_NODES:
                break
        
        print(f"[{request_id}] Total unique nodes for synthesis: {len(unique_nodes_for_synthesis)}")


        # 3️⃣ Synthesize Final Response
        # The hardcoded response for "no nodes found" has been removed.
        # The LLM (via ResponseSynthesizer) will now always be called.
        # If unique_nodes_for_synthesis is empty, the LLM should use the system_prompt
        # to formulate a response indicating no information was found.

        if not unique_nodes_for_synthesis:
            print(f"[{request_id}] No unique nodes found. LLM will synthesize response based on system prompt and no context.")
        else:
            print(f"[{request_id}] Synthesizing response with {len(unique_nodes_for_synthesis)} nodes.")
            
        query_bundle = QueryBundle(query_str=req.message)
        
        synthesized_response_obj = await asyncio.to_thread(
            response_synthesizer.synthesize,
            query_bundle,
            nodes=unique_nodes_for_synthesis # This list might be empty
        )
        
        final_answer = synthesized_response_obj.response
        final_sources = extract_sources(synthesized_response_obj) # Will be [] if unique_nodes_for_synthesis is []

        # processing_time = time.time() - t_start
        # decision_log_msg = f"SYNTHESIZED_FROM_TOP_{len(unique_nodes_for_synthesis)}_NODES"
        # logger.info("[%s] Decision: %s  |  Time: %.2fs", request_id, decision_log_msg, processing_time) # Example for uvicorn logger
        print(f"[{request_id}] Sending response: '{final_answer[:100]}...' with {len(final_sources)} sources.")
        return ChatResponse(response=final_answer, sources=final_sources)

    except Exception as e:
        print(f"[{request_id}] CRITICAL Unhandled error in /chat endpoint: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

# ------------------------------------------------------------------------------
# Local Run (for development)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"Starting Uvicorn server on http://0.0.0.0:{port}")
    # To run with Uvicorn and enable auto-reload for development:
    # uvicorn main:app --reload --host 0.0.0.0 --port 8000
    # (Assuming your file is named main.py)
    uvicorn.run(app, host="0.0.0.0", port=port)
