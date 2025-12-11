"""
Allora Topic Wizard tools for LlamaIndex FunctionAgent

Tools are dynamically fetched from the wizard backend at startup.
This keeps the RAG repo clean and ensures tools are always in sync.
"""

import logging
import json
from typing import Annotated, List, Literal, Optional, Callable
import httpx
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


async def fetch_tool_schema() -> Optional[dict]:
    """Fetch tool schema from the wizard backend."""
    global _cached_schema
    if _cached_schema is not None:
        return _cached_schema

    config = get_config()
    url = f"{config.wizard.api_url}/api/tools/schema"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            if data.get("success"):
                _cached_schema = data
                logger.info(f"Fetched {len(data.get('tools', []))} tool schemas from wizard backend")
                return data
    except Exception as e:
        logger.warning(f"Failed to fetch tool schema from wizard backend: {e}")
    return None


def fetch_tool_schema_sync() -> Optional[dict]:
    """Synchronous version for startup."""
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


async def make_request(endpoint: str, params: Optional[dict] = None) -> str:
    """Make HTTP GET request to wizard API and return formatted response."""
    config = get_config()
    url = f"{config.wizard.api_url}{endpoint}"

    headers = {}
    if config.wizard.api_key:
        headers["Authorization"] = f"Bearer {config.wizard.api_key}"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                params=params,
                headers=headers if headers else None,
                timeout=config.wizard.timeout
            )
            response.raise_for_status()
            data = response.json()

            if data.get("success"):
                return json.dumps(data.get("data", data), indent=2)
            else:
                return f"Error: {data.get('message', 'Unknown error')}"

    except httpx.ConnectError:
        logger.error(f"Connection failed to wizard API: {url}")
        return SERVER_UNAVAILABLE_MESSAGE
    except httpx.ConnectTimeout:
        logger.error(f"Connection timeout to wizard API: {url}")
        return SERVER_UNAVAILABLE_MESSAGE
    except httpx.TimeoutException:
        logger.error(f"Request timeout calling wizard API: {url}")
        return "Error: Request timed out. Please try again."
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error calling wizard API: {e}")
        if e.response.status_code == 401:
            return "Error: Authentication failed."
        if e.response.status_code == 403:
            return "Error: Access forbidden."
        if e.response.status_code >= 500:
            return SERVER_UNAVAILABLE_MESSAGE
        return f"Error: HTTP {e.response.status_code} - {e.response.text}"
    except Exception as e:
        logger.error(f"Error calling wizard API: {e}")
        error_str = str(e).lower()
        if any(x in error_str for x in ["connection", "refused", "unreachable"]):
            return SERVER_UNAVAILABLE_MESSAGE
        return f"Error: {str(e)}"


def create_tool_function(schema: dict) -> Callable:
    """Create an async function from a tool schema."""
    endpoint_template = schema["endpoint"]
    params_schema = schema.get("params", {})
    extra_params = schema.get("extra_params", {})

    async def tool_fn(**kwargs) -> str:
        endpoint = endpoint_template
        query_params = dict(extra_params) if extra_params else {}

        for param_name, param_config in params_schema.items():
            value = kwargs.get(param_name)

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

        return await make_request(endpoint, query_params if query_params else None)

    return tool_fn


def build_function_signature(schema: dict, type_definitions: dict) -> dict:
    """Build parameter annotations for a tool function."""
    params_schema = schema.get("params", {})
    annotations = {}

    # Build type mapping from server-provided types
    type_mapping = {"str": str, "bool": bool}
    for type_name, options in type_definitions.items():
        if isinstance(options, list):
            type_mapping[type_name] = Literal[tuple(options)]

    for param_name, param_config in params_schema.items():
        param_type = type_mapping.get(param_config["type"], str)
        description = param_config.get("description", param_name)

        if param_config.get("optional") or "default" in param_config:
            annotations[param_name] = Annotated[Optional[param_type], description]
        else:
            annotations[param_name] = Annotated[param_type, description]

    return annotations


def build_pydantic_model(schema: dict, type_definitions: dict):
    """Build a Pydantic model for tool parameters."""
    from pydantic import BaseModel, Field, create_model
    from typing import Optional as Opt

    params_schema = schema.get("params", {})
    if not params_schema:
        return None

    type_mapping = {"str": str, "bool": bool}
    for type_name, options in type_definitions.items():
        if isinstance(options, list):
            type_mapping[type_name] = Literal[tuple(options)]

    fields = {}
    for param_name, param_config in params_schema.items():
        param_type = type_mapping.get(param_config["type"], str)
        description = param_config.get("description", param_name)
        default = param_config.get("default", ...)

        if param_config.get("optional"):
            fields[param_name] = (Opt[param_type], Field(default=None, description=description))
        elif default != ...:
            fields[param_name] = (param_type, Field(default=default, description=description))
        else:
            fields[param_name] = (param_type, Field(..., description=description))

    return create_model(f"{schema['name']}_params", **fields)


def create_wizard_tools() -> List[BaseTool]:
    """Create wizard tools by fetching schema from backend."""
    schema_data = fetch_tool_schema_sync()

    if not schema_data:
        logger.warning("Could not fetch tool schema - wizard tools will not be available")
        return []

    tools = []
    tool_schemas = schema_data.get("tools", [])
    type_definitions = schema_data.get("types", {})

    for schema in tool_schemas:
        fn = create_tool_function(schema)
        fn.__name__ = schema["name"]
        fn.__doc__ = schema["description"]

        # Build Pydantic model for proper parameter schema
        fn_schema = build_pydantic_model(schema, type_definitions)

        tool = FunctionTool.from_defaults(
            fn=fn,
            name=schema["name"],
            description=schema["description"],
            fn_schema=fn_schema,
        )
        tools.append(tool)

    logger.info(f"Created {len(tools)} wizard tools from server schema")
    return tools
