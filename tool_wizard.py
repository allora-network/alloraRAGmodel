"""
Allora Topic Wizard tools for LlamaIndex FunctionAgent

Provides natural language interface to the Wizard API using GPT function calling.
Tool schemas are fetched dynamically from the wizard backend.
"""

import json
import logging
import re
from typing import List, Optional

import httpx
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.core.tools import FunctionTool, BaseTool

from config import get_config

logger = logging.getLogger("uvicorn.error")

# Cached data
_schema_cache: Optional[dict] = None
_llm_client: Optional[LlamaOpenAI] = None


def _get_llm() -> LlamaOpenAI:
    """Get or create shared LlamaIndex OpenAI client."""
    global _llm_client
    if _llm_client is None:
        config = get_config()
        _llm_client = LlamaOpenAI(
            model=config.agent.model,
            temperature=config.agent.temperature,
            reuse_client=config.agent.reuse_client,
        )
    return _llm_client


def _fetch_schema() -> Optional[dict]:
    """Fetch and cache tool schema from wizard backend."""
    global _schema_cache
    if _schema_cache is not None:
        return _schema_cache

    config = get_config()
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{config.wizard.api_url}/api/tools/schema")
            resp.raise_for_status()
            data = resp.json()
            if data.get("success"):
                _schema_cache = data
                logger.info(f"Loaded {len(data.get('tools', []))} wizard tools")
                return data
    except Exception as e:
        logger.warning(f"Failed to fetch wizard schema: {e}")
    return None


def _call_api(endpoint: str, params: Optional[dict] = None) -> dict:
    """Make API call to wizard backend."""
    config = get_config()
    headers = {"Authorization": f"Bearer {config.wizard.api_key}"} if config.wizard.api_key else {}

    try:
        with httpx.Client(timeout=config.wizard.timeout) as client:
            resp = client.get(f"{config.wizard.api_url}{endpoint}", params=params, headers=headers)
            resp.raise_for_status()
            return resp.json()
    except httpx.ConnectError:
        return {"success": False, "error": "Wizard service unavailable"}
    except httpx.TimeoutException:
        return {"success": False, "error": "Request timed out"}
    except httpx.HTTPStatusError as e:
        return {"success": False, "error": f"HTTP {e.response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _build_openai_tools(schema: dict) -> list:
    """Convert wizard schema to OpenAI function calling format."""
    types = schema.get("types", {})
    tools = []

    for tool in schema.get("tools", []):
        properties = {}
        required = []

        for name, cfg in tool.get("params", {}).items():
            ptype = cfg.get("type", "str")
            prop = {"description": cfg.get("description", name)}

            if ptype in types:
                prop["type"] = "string"
                prop["enum"] = types[ptype]
            elif ptype == "bool":
                prop["type"] = "boolean"
            else:
                prop["type"] = "string"

            properties[name] = prop
            if not cfg.get("optional") and "default" not in cfg:
                required.append(name)

        tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": {"type": "object", "properties": properties, "required": required}
            }
        })

    return tools


def _execute_tool(name: str, args: dict, schema: dict) -> dict:
    """Execute a tool call and return result."""
    tool_def = next((t for t in schema.get("tools", []) if t["name"] == name), None)
    if not tool_def:
        return {"success": False, "error": f"Unknown tool: {name}"}

    endpoint = tool_def["endpoint"]
    params = dict(tool_def.get("extra_params", {}))

    for pname, pcfg in tool_def.get("params", {}).items():
        value = args.get(pname)
        if value is None:
            value = pcfg.get("default")
        if value is None:
            continue

        if pcfg.get("path"):
            endpoint = endpoint.replace(f"{{{pname}}}", str(value))
        else:
            key = pcfg.get("query_key", pname)
            params[key] = "true" if value is True else "false" if value is False else value

    return _call_api(endpoint, params if params else None)


SYSTEM_PROMPT = """You are an Allora Network assistant. Use the available tools to answer questions about topics, services, and infrastructure.

Guidelines:
- Default to "testnet" unless user specifies "mainnet"
- Topic IDs are numeric strings (e.g., "1", "70")
- Addresses start with "allo1..."
- Be concise and helpful"""


async def wizard_query(query: str) -> str:
    """
    Query the Allora Topic Wizard for blockchain and service information.

    Capabilities:
    - Topic info: details, stake, activity, fee revenue, inferences
    - Whitelist: check status, list all whitelisted workers/reputers
    - Registration: worker/reputer status and node info
    - Scores: inferer, forecaster, reputer scores and stakes
    - Services: OSM configs, wallets, Kubernetes status

    Examples:
    - "What is topic 5 on testnet?"
    - "List all whitelisted workers on topic 70"
    - "Is topic 14 active on mainnet?"
    """
    schema = _fetch_schema()
    if not schema:
        return "Wizard service unavailable. Please ensure the backend is running."

    llm = _get_llm()
    tools = _build_openai_tools(schema)

    try:
        # Use the underlying OpenAI client from LlamaIndex (with reuse_client)
        client = llm._get_client()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ]

        # Let GPT decide which tools to call
        response = client.chat.completions.create(
            model=llm.model,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        # Process tool calls iteratively
        while response.choices[0].message.tool_calls:
            msg = response.choices[0].message
            messages.append(msg)

            for call in msg.tool_calls:
                args = json.loads(call.function.arguments)
                result = _execute_tool(call.function.name, args, schema)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result)
                })

            response = client.chat.completions.create(
                model=llm.model,
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )

        return response.choices[0].message.content or "No response generated."

    except Exception as e:
        logger.error(f"Wizard query error: {e}")
        # Fallback to simple query
        return _simple_query(query, schema)


def _simple_query(query: str, schema: dict) -> str:
    """Fallback for when LLM fails - basic keyword matching."""
    q = query.lower()
    network = "mainnet" if "mainnet" in q else "testnet"

    # Extract topic ID
    match = re.search(r'topic\s*(\d+)', q)
    if match:
        topic_id = match.group(1)
        result = _call_api(f"/api/sdk/query/topic/{topic_id}", {"network": network})
        if result.get("success"):
            return f"Topic {topic_id} ({network}):\n```json\n{json.dumps(result, indent=2)}\n```"
        return f"Error: {result.get('error', 'Unknown error')}"

    # List services
    if "list" in q and ("forecaster" in q or "reputer" in q):
        stype = "forecaster" if "forecaster" in q else "reputer"
        result = _call_api("/api/osm/config", {"network": network, "purpose": stype})
        if result.get("success"):
            return f"{stype.title()}s ({network}):\n```json\n{json.dumps(result, indent=2)}\n```"

    return "Could not understand query. Try: 'What is topic 5?' or 'List forecasters on testnet'"


# Export the tool
wizard_tool = FunctionTool.from_defaults(
    async_fn=wizard_query,
    name="wizard_query",
    description="""Query the Allora Topic Wizard for blockchain and service information.

Use for: topic details, whitelist status, worker/reputer info, scores, OSM configs, K8s status.

Examples:
- "What is topic 5 on testnet?"
- "List all whitelisted workers on topic 70"
- "Is allo1xyz whitelisted on topic 5?"
- "Get inferer score for allo1abc on topic 10"
"""
)


def create_wizard_tools() -> List[BaseTool]:
    """Create wizard tools for the agent."""
    return [wizard_tool]
