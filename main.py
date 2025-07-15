import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from fastapi import FastAPI, Response, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from llm import Agent
from slack import process_slack_message
from exceptions import AlloraAgentError, SlackIntegrationError, ConfigurationError
from config import get_config

logger = logging.getLogger("uvicorn.error") # -> debugging purposes

SESSION_COOKIE_NAME = 'session_id'

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

# ── Configuration Setup ──────────────────────────────────────────────
# Initialize configuration (this will validate environment variables)
config = get_config()

# ── FastAPI setup ───────────────────────────────────────────────────────────
app = FastAPI(debug=config.server.debug, default_response_class=PrettyJSONResponse)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.server.allowed_origins,
    allow_credentials=config.server.allow_credentials,
    allow_methods=config.server.allowed_methods,
    allow_headers=config.server.allowed_headers,
)

@app.middleware("http")
async def ensure_session_id(request: Request, call_next):
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    if not session_id:
        session_id = str(uuid.uuid4())
        response = await call_next(request)
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=session_id,
            httponly=True,   # Best practice
            samesite="Lax",  # Or "Strict" or "None"
            secure=True      # True if using HTTPS
        )
        return response
    else:
        # Already has session cookie
        return await call_next(request)


# ── Health Check ──────────────────────────────────────────────────────
@app.get("/")
async def root():
    logger.info(f"/ (health check)")
    return "ok"


# ── Raw chat endpoint ──────────────────────────────────────────────────────
@app.post("/chat")
async def chat_endpoint(request: Request, req: ChatRequest):
    logger.info(f"/chat endpoint hit")
    t0 = time.time()

    request_id = id(req)  # Simple request ID for tracking
    logger.info(f"[{request_id}] Starting chat request: '{req.message[:50]}{'...' if len(req.message) > 50 else ''}'")

    try:
        session_id = request.cookies.get(SESSION_COOKIE_NAME)
        if not session_id:
            raise Exception('no session id')

        docs_agent = Agent(
            session_id=session_id,
            index_names=["alloradocs", "allora_chain", "allora_production"],
            max_tokens=16384,
        )

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
@app.post("/slack")
async def slack_endpoint(request: Request):
    try:
        json_body = await request.json()
        request_id = id(request)
        logger.info(f"/slack [{request_id}] {json_body.get('type', 'unknown')}")
        
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
                channel = event_data.get("channel")
                if channel is None:
                    raise Exception("no channel specified")

                slack_agent = Agent(
                    session_id=channel + "." + event_data.get("ts"),
                    index_names=["alloradocs", "allora_chain", "allora_production"],
                    max_tokens=16384,
                )

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


# ── Generated images ──────────────────────────────────────────────────────
IMAGE_DIR = Path("static/images")
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# Mount the directory to serve images at /images
app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")





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
