# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is the Allora RAG Information Bot - an agentic RAG chatbot powered by GPT-4 that answers questions about Allora Labs using content from documentation, research papers, and codebase. The bot features automatic chart generation capabilities using matplotlib for precise data visualization when visual content would enhance responses. The bot is accessible via Slack, Discord, and documentation websites.

## Architecture

The system uses a multi-index approach with LlamaCloud vector databases:
- **Vector Storage**: LlamaCloud with 3072-dimensional vectors using OpenAI `text-embedding-3-large`
- **LLM**: GPT-4o-mini for query processing with custom system prompts
- **API**: FastAPI server with async endpoints
- **Agent System**: Custom Agent class that queries multiple indices and uses score-based selection
- **Chart Generation**: Matplotlib-based programmatic chart generation for precise data visualization
- **Multimodal Responses**: Text + chart responses when visual aids enhance understanding

### Key Components

- `main.py`: FastAPI application with `/chat` and `/slack` endpoints, Slack chart upload functionality
- `llm.py`: Core Agent class handling multi-index queries, response selection, and chart generation
- `sysprompt.py`: System prompt for "Allie" the Allora assistant
- `__init__.py`: Package initialization

### Chart Generation Features

- **Smart Detection**: Automatically detects when queries would benefit from data visualization
- **RAG-Driven Charts**: Analyzes actual retrieved content to create precise, data-accurate visualizations
- **Data Extraction**: Extracts numbers, metrics, entities, and relationships from RAG responses
- **Programmatic Generation**: Uses matplotlib to create professional charts with exact data representation
- **Chart Types**: Reward distribution bars, metrics dashboards, workflow diagrams, comparison charts
- **Slack Integration**: Uploads generated charts directly to Slack channels/DMs
- **Temporary Storage**: Manages temporary chart files with automatic cleanup
- **Fallback Handling**: Graceful degradation when chart generation fails

### Query Processing Flow

1. User submits question via POST endpoint
2. Agent queries multiple LlamaCloud indices simultaneously (`allora_chain`, `allora_production`, `alloradocs`)
3. Results scored by cosine similarity
4. If best score > 0.85 with 0.10+ margin over second-best, use that response
5. Otherwise, use LLM selector to choose best index and re-query
6. **Content Analysis**: Extracts visualizable data (numbers, metrics, entities) from RAG response
7. **Visualization Type Detection**: Determines optimal chart type (rewards, metrics, workflow, comparison)
8. **Python Code Generation**: Creates matplotlib code based on actual retrieved data
9. **Chart Execution**: Executes Python code to generate precise data visualizations
10. **Multimodal Response**: Returns text + optional chart path

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run development server
python main.py

# Run with Docker
docker build -t allora-rag .
docker run -p 8000:8000 allora-rag
```

## Environment Variables

Required environment variables:
- `LLAMA_CLOUD_API_KEY`: LlamaCloud API key
- `LLAMA_CLOUD_ORG_ID`: LlamaCloud organization ID  
- `OPENAI_API_KEY`: OpenAI API key (required for LLM)
- `SLACK_BOT_TOKEN`: Slack bot token (starts with xoxb-) for Slack integration
- `PORT`: Server port (default: 8000)

### Slack Configuration for Charts

**Required OAuth Scopes for Chart Upload:**
- `files:write`: Upload and share charts
- `app_mentions:read`: Respond to mentions
- `chat:write`: Send messages with chart attachments

## API Endpoints

- `GET /`: Health check
- `POST /chat`: Main chat endpoint (requires `{"message": "string"}`)
- `POST /chat/debug`: Debug endpoint with detailed source extraction logging
- `POST /slack`: Slack Events API webhook endpoint

### Slack Integration

The `/slack` endpoint handles Slack Events API webhooks for:
1. **Direct Messages**: Users can message the bot directly (`message.im` events)
2. **Channel Mentions**: Users can mention the bot with @ in channels (`app_mention` events)
3. **Image Uploads**: Automatically uploads generated images to appropriate channels/threads

**Required Slack App Configuration:**
- **OAuth Scopes**: `app_mentions:read`, `chat:write`, `channels:read`, `files:write`
- **Event Subscriptions**: Subscribe to `message.im` and `app_mention` events
- **Request URL**: `https://your-domain.com/slack`

**Chart Generation Triggers:**
The bot automatically generates charts when queries contain keywords like:
- "diagram", "chart", "graph", "visual", "plot"
- "show me", "draw", "create", "visualization"
- "distribution", "comparison", "metrics", "performance"

## Testing

Test the chatbot server:

```python
import requests

url = "http://localhost:8000/chat"
payload = {"message": "What makes Allora's reward distribution different?"}
response = requests.post(url, json=payload)
print(response.json())
```

### Debugging Source Extraction

Use the debug endpoint to troubleshoot source extraction issues:

```python
import requests

url = "http://localhost:8000/chat/debug"
payload = {"message": "What is Allora?"}
response = requests.post(url, json=payload)
print(response.json())

# Check server logs for detailed debugging information including:
# - Query engine scores and selection logic
# - Source node metadata inspection
# - Metadata field analysis
# - Response structure details
# - Image generation decisions and processes
```

### Testing Chart Generation

Test chart generation capabilities:

```python
import requests

# Test chart generation trigger
url = "http://localhost:8000/chat"
payload = {"message": "Show me a chart of reward distribution to workers"}
response = requests.post(url, json=payload)
result = response.json()

print("Response:", result["response"])
print("Sources:", result["sources"])
print("Chart generated:", result["image_generated"])
if result["image_generated"]:
    print("Chart path:", result["image_path"])
```

## Configuration

- **Similarity threshold**: 0.85 (THRESH in llm.py:22)
- **Score margin**: 0.10 (MARGIN in llm.py:23)
- **Top-k similarity**: 5 documents per query
- **Max tokens**: 1000 for responses, 100 for tool selection
- **Temperature**: 0 (deterministic responses)