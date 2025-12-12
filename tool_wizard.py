"""
Allora Topic Wizard tools for LlamaIndex FunctionAgent

Uses Claude Opus 4.5 for intelligent tool selection and execution.
Tool schemas are fetched from the wizard backend at startup.
"""

import logging
import json
from typing import List, Optional
import httpx
import anthropic
from llama_index.core.tools import FunctionTool, BaseTool
from config import get_config

logger = logging.getLogger("uvicorn.error")

SERVER_UNAVAILABLE_MESSAGE = (
    "The Allora Topic Wizard service is currently unavailable. "
    "This tool requires the wizard backend to be running. "
    "Please try again later or contact the team if the issue persists."
)

# Cache for the fetched schema
_cached_schema: Optional[dict] = None
_anthropic_client: Optional[anthropic.Anthropic] = None


def get_anthropic_client() -> Optional[anthropic.Anthropic]:
    """Get or create Anthropic client."""
    global _anthropic_client
    if _anthropic_client is not None:
        return _anthropic_client

    config = get_config()
    if not config.wizard.anthropic_api_key:
        logger.warning("ANTHROPIC_API_KEY not set - wizard will use direct API calls without Claude reasoning")
        return None

    _anthropic_client = anthropic.Anthropic(api_key=config.wizard.anthropic_api_key)
    return _anthropic_client


def fetch_tool_schema_sync() -> Optional[dict]:
    """Fetch tool schema from the wizard backend."""
    global _cached_schema
    if _cached_schema is not None:
        return _cached_schema

    config = get_config()
    url = f"{config.wizard.api_url}/api/tools/schema"

    try:
        with httpx.Client() as client:
            response = client.get(url, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            if data.get("success"):
                _cached_schema = data
                logger.info(f"Fetched {len(data.get('tools', []))} tool schemas from wizard backend")
                return data
    except Exception as e:
        logger.warning(f"Failed to fetch tool schema from wizard backend: {e}")
    return None


def make_api_call(endpoint: str, params: Optional[dict] = None) -> dict:
    """Make HTTP GET request to wizard API and return response dict."""
    config = get_config()
    url = f"{config.wizard.api_url}{endpoint}"

    headers = {}
    if config.wizard.api_key:
        headers["Authorization"] = f"Bearer {config.wizard.api_key}"

    try:
        with httpx.Client() as client:
            response = client.get(
                url,
                params=params,
                headers=headers if headers else None,
                timeout=config.wizard.timeout
            )
            response.raise_for_status()
            return response.json()

    except httpx.ConnectError:
        return {"success": False, "error": "Connection failed - wizard service unavailable"}
    except httpx.TimeoutException:
        return {"success": False, "error": "Request timed out"}
    except httpx.HTTPStatusError as e:
        return {"success": False, "error": f"HTTP {e.response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def build_claude_tools(schema_data: dict) -> list:
    """Build Claude tool definitions from schema data."""
    tools = []
    type_definitions = schema_data.get("types", {})

    for schema in schema_data.get("tools", []):
        properties = {}
        required = []

        for param_name, param_config in schema.get("params", {}).items():
            param_type = param_config.get("type", "str")

            # Map type to JSON Schema
            if param_type in type_definitions:
                prop = {
                    "type": "string",
                    "enum": type_definitions[param_type],
                    "description": param_config.get("description", param_name)
                }
            elif param_type == "bool":
                prop = {
                    "type": "boolean",
                    "description": param_config.get("description", param_name)
                }
            else:
                prop = {
                    "type": "string",
                    "description": param_config.get("description", param_name)
                }

            properties[param_name] = prop

            if not param_config.get("optional") and "default" not in param_config:
                required.append(param_name)

        tools.append({
            "name": schema["name"],
            "description": schema["description"],
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        })

    return tools


def execute_tool_call(tool_name: str, tool_input: dict, schema_data: dict) -> dict:
    """Execute a tool call based on schema."""
    # Find the tool schema
    tool_schema = None
    for schema in schema_data.get("tools", []):
        if schema["name"] == tool_name:
            tool_schema = schema
            break

    if not tool_schema:
        return {"success": False, "error": f"Unknown tool: {tool_name}"}

    # Build endpoint and params
    endpoint = tool_schema["endpoint"]
    params_schema = tool_schema.get("params", {})
    extra_params = tool_schema.get("extra_params", {})
    query_params = dict(extra_params) if extra_params else {}

    for param_name, param_config in params_schema.items():
        value = tool_input.get(param_name)

        if value is None and param_config.get("optional"):
            continue
        if value is None and "default" in param_config:
            value = param_config["default"]
        if value is None:
            continue

        if param_config.get("path"):
            endpoint = endpoint.replace(f"{{{param_name}}}", str(value))
        else:
            query_key = param_config.get("query_key", param_name)
            if isinstance(value, bool):
                value = "true" if value else "false"
            query_params[query_key] = value

    return make_api_call(endpoint, query_params if query_params else None)


async def wizard_query(query: str) -> str:
    """
    Query the Allora Topic Wizard using Claude Opus 4.5.

    This tool uses Claude to intelligently determine which wizard API calls
    to make based on your natural language query. It can:

    Topic Information:
    - Get topic details (metadata, stake, activity status, fee revenue)
    - Check if a topic exists
    - Get network inference and latest inferences

    Whitelist Operations:
    - Check if whitelist is enabled for workers/reputers
    - Check if specific addresses are whitelisted
    - List ALL whitelisted workers or reputers for a topic

    Registration & Scores:
    - Check worker/reputer registration status
    - Get worker/reputer node info
    - Get inferer, forecaster, and reputer scores
    - Get reputer stake in a topic

    Services & Infrastructure:
    - List services (forecasters, reputers) and their configurations
    - Query OSM configurations and wallet addresses
    - Inspect Kubernetes deployments, configmaps, and pod status

    Args:
        query: Natural language query about Allora topics, services, or configurations.
               Examples:
               - "What is topic 5 on testnet?"
               - "Is topic 14 active on mainnet?"
               - "List all forecasters serving topic 10"
               - "What's the total stake in topic 1 on mainnet?"
               - "List all whitelisted workers on topic 70"
               - "How many reputers are whitelisted on topic 5?"
               - "What's the inferer score for allo1xyz on topic 10?"

    Returns:
        Detailed response based on the wizard API data.
    """
    config = get_config()
    schema_data = fetch_tool_schema_sync()

    if not schema_data:
        return SERVER_UNAVAILABLE_MESSAGE

    client = get_anthropic_client()

    # If no Anthropic client, fall back to simple keyword matching
    if not client:
        return await _fallback_query(query, schema_data)

    # Build tools for Claude
    claude_tools = build_claude_tools(schema_data)

    system_prompt = """You are an expert assistant for the Allora Network. Your role is to help users query information about topics, services, and configurations using the available tools.

When the user asks a question:
1. Determine which tool(s) to call based on their query
2. Call the appropriate tools with the correct parameters
3. Analyze the results and provide a clear, helpful response

Important notes:
- Default to "testnet" network unless the user specifies "mainnet"
- Topic IDs are numeric strings (e.g., "1", "5", "14")
- Addresses start with "allo1..."
- For listing services, use list_services with the appropriate service_type (reputer or forecaster)

Available tool categories:
- Topic queries: get_topic, topic_exists, is_topic_active, get_topic_stake, get_topic_fee_revenue
- Whitelist queries: is_worker/reputer_whitelist_enabled, is_worker/reputer_whitelisted, get_whitelisted_workers, get_whitelisted_reputers
- Registration: is_worker/reputer_registered, get_worker/reputer_node_info
- Scores & stakes: get_inferer_score, get_forecaster_score, get_reputer_score, get_reputer_stake_in_topic
- Inferences: get_latest_inferences, get_network_inference
- Services: list_services, get_osm_config, get_osm_wallets, get_osm_topic_details
- Kubernetes: get_k8s_deployments, get_k8s_configmaps, get_k8s_pod_status

Be concise but thorough in your responses. If data is missing or there's an error, explain what happened."""

    messages = [{"role": "user", "content": query}]

    try:
        # First Claude call - let it decide what tools to use
        response = client.messages.create(
            model=config.wizard.anthropic_model,
            max_tokens=4096,
            system=system_prompt,
            tools=claude_tools,
            messages=messages
        )

        # Process tool calls in a loop until Claude is done
        while response.stop_reason == "tool_use":
            # Extract tool uses from response
            tool_results = []
            assistant_content = response.content

            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    tool_id = block.id

                    logger.debug(f"Claude calling tool: {tool_name} with {tool_input}")

                    # Execute the tool
                    result = execute_tool_call(tool_name, tool_input, schema_data)

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": json.dumps(result, indent=2)
                    })

            # Continue conversation with tool results
            messages = [
                {"role": "user", "content": query},
                {"role": "assistant", "content": assistant_content},
                {"role": "user", "content": tool_results}
            ]

            response = client.messages.create(
                model=config.wizard.anthropic_model,
                max_tokens=4096,
                system=system_prompt,
                tools=claude_tools,
                messages=messages
            )

        # Extract final text response
        final_response = ""
        for block in response.content:
            if hasattr(block, "text"):
                final_response += block.text

        return final_response if final_response else "I couldn't generate a response. Please try rephrasing your question."

    except anthropic.APIError as e:
        logger.error(f"Anthropic API error: {e}")
        return f"Error communicating with AI service: {str(e)}"
    except Exception as e:
        logger.error(f"Error in wizard_query: {e}")
        return f"An error occurred: {str(e)}"


async def _fallback_query(query: str, schema_data: dict) -> str:
    """Simple fallback when Anthropic is not available."""
    query_lower = query.lower()

    # Try to extract topic ID
    import re
    topic_match = re.search(r'topic\s*(\d+)', query_lower)
    topic_id = topic_match.group(1) if topic_match else None

    # Determine network
    network = "mainnet" if "mainnet" in query_lower else "testnet"

    if topic_id:
        # Get topic info
        result = make_api_call(f"/api/sdk/query/topic/{topic_id}", {"network": network})
        if result.get("success"):
            return f"Topic {topic_id} on {network}:\n```json\n{json.dumps(result.get('data', result), indent=2)}\n```"
        else:
            return f"Error fetching topic {topic_id}: {result.get('error', 'Unknown error')}"

    if "list" in query_lower and ("forecaster" in query_lower or "reputer" in query_lower):
        service_type = "forecaster" if "forecaster" in query_lower else "reputer"
        result = make_api_call("/api/osm/config", {"network": network, "purpose": service_type})
        if result.get("success"):
            return f"{service_type.title()}s on {network}:\n```json\n{json.dumps(result.get('data', result), indent=2)}\n```"

    return "I couldn't understand your query. Please try asking about a specific topic (e.g., 'What is topic 5 on testnet?') or listing services (e.g., 'List forecasters on testnet')."


# Create the wizard tool for the main agent
wizard_tool = FunctionTool.from_defaults(
    async_fn=wizard_query,
    name="wizard_query",
    description="""Query the Allora Topic Wizard for blockchain and service information.

Use this tool to get information about:
- Topic details (metadata, stake, activity, fee revenue, existence)
- Whitelist status (check individual addresses OR list ALL whitelisted workers/reputers)
- Registration status and node info for workers/reputers
- Scores (inferer, forecaster, reputer) and reputer stakes
- Service configurations (forecasters, reputers) from OSM
- Kubernetes infrastructure (deployments, pods, configmaps)
- Network inference data and latest inferences

Examples:
- "What is topic 5 on testnet?"
- "Is topic 14 active on mainnet?"
- "List forecasters serving topic 10"
- "What's the total stake in topic 1?"
- "Is address allo1xyz whitelisted as a worker on topic 5?"
- "List all whitelisted workers on topic 70"
- "How many reputers are whitelisted on topic 5?"
- "Get the inferer score for allo1abc on topic 10"
- "What pods are running for topic 15?"
"""
)


def create_wizard_tools() -> List[BaseTool]:
    """Create wizard tools for the agent."""
    return [wizard_tool]
