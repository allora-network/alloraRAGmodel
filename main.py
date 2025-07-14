import asyncio
import json
import logging
import os
import time
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from fastapi import FastAPI, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from llm import Agent
from slack import process_slack_message
from exceptions import AlloraAgentError, SlackIntegrationError, ConfigurationError

logger = logging.getLogger("uvicorn.error") # -> debugging purposes


# ── Requests/responses ───────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    sources: List[str]

class PrettyJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            separators=(",", ": "),
        ).encode("utf-8")

# ── Environment Configuration ──────────────────────────────────────────────
DEBUG_MODE = os.getenv("DEBUG", "false").lower() == "true"
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")

# Validate required environment variables
REQUIRED_ENV_VARS = [
    "LLAMA_CLOUD_API_KEY",
    "LLAMA_CLOUD_ORG_ID", 
    "OPENAI_API_KEY",
    "SLACK_BOT_TOKEN"
]

missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
if missing_vars:
    raise ConfigurationError(f"Missing required environment variables: {', '.join(missing_vars)}")

# ── FastAPI setup ───────────────────────────────────────────────────────────
app = FastAPI(debug=DEBUG_MODE, default_response_class=PrettyJSONResponse)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Health Check ──────────────────────────────────────────────────────
@app.get("/")
async def root():
    logger.info(f"/ (health check)")
    return "ok"


# ── Raw chat endpoint ──────────────────────────────────────────────────────
docs_agent = Agent(index_names=["alloradocs"], max_tokens=2000, enable_chart_generation=True)

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    logger.info(f"/chat endpoint hit")
    t0 = time.time()

    request_id = id(req)  # Simple request ID for tracking
    logger.info(f"[{request_id}] Starting chat request: '{req.message[:50]}{'...' if len(req.message) > 50 else ''}'")

    try:
        answer, sources, image_paths = await docs_agent.answer_allora_query(request_id, req.message)

        total_time = time.time() - t0
        logger.info(f"[{request_id}] Request completed - Total time: {total_time:.2f}s")
        
        # Return response (will be pretty-printed automatically)
        response_data = {
            "response": answer,
            "sources": sources,
            "query_time": f"{total_time:.2f}s",
            "request_id": str(request_id)
        }
        
        # Add image info if available
        response_data["image_paths"] = image_paths
        
        return response_data
    
    except AlloraAgentError as e:
        logger.error(f"[{request_id}] Agent error: {str(e)}")
        return {"error": "I encountered an issue processing your request. Please try again.", "request_id": str(request_id)}
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error: {str(e)}")
        return {"error": "An unexpected error occurred. Please try again later.", "request_id": str(request_id)}


# ── Slack bot ──────────────────────────────────────────────────────
slack_agent = Agent(index_names=["alloradocs", "allora_chain", "allora_production"], max_tokens=16384, enable_chart_generation=True)

@app.post("/slack")
async def slack_endpoint(request: Request):
    """Handle Slack Events API webhooks for DMs and mentions"""
    
    try:
        json_body = await request.json()
        request_id = id(request)
        logger.info(f"/slack [{request_id}] Received Slack request: {json_body.get('type', 'unknown')}")
        
        # Handle URL verification challenge
        if json_body.get("type") == "url_verification":
            challenge = json_body.get("challenge")
            logger.info(f"[{request_id}] URL verification challenge received")
            return Response(content=challenge, headers={"Content-Type": "text/plain"}, status_code=200)
        
        # Handle event callbacks
        if json_body.get("type") == "event_callback":
            event_data = json_body.get("event", {})
            event_type = event_data.get("type")
            
            # Ignore bot messages to prevent loops
            if event_data.get("bot_id") or event_data.get("subtype") == "bot_message":
                logger.info(f"[{request_id}] Ignoring bot message")
                return Response(status_code=200)
            
            # Handle direct messages (message.im) and mentions (app_mention)
            if event_type in ["message", "app_mention"]:
                # Start async processing (don't await to return 200 quickly)
                asyncio.create_task(process_slack_message(event_data, str(request_id), slack_agent))
                return Response(status_code=200)
            
            logger.info(f"[{request_id}] Unhandled event type: {event_type}")
            return Response(status_code=200)
        
        logger.warning(f"[{request_id}] Unhandled request type: {json_body.get('type')}")
        return Response(status_code=200)
        
    except SlackIntegrationError as e:
        logger.error(f"Slack integration error: {str(e)}")
        return Response(status_code=400)
    except AlloraAgentError as e:
        logger.error(f"Agent error processing Slack request: {str(e)}")
        return Response(status_code=500)
    except Exception as e:
        logger.error(f"Unexpected error processing Slack request: {str(e)}")
        return Response(status_code=500)





if __name__ == "__main__":
    # env vars
    os.environ["LLAMA_CLOUD_API_KEY"] = os.getenv("LLAMA_CLOUD_API_KEY")
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    # logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
