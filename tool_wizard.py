"""
Wizard tools via MCP server integration.

Uses llama-index-tools-mcp to run the MCP server and get tools directly.
This replaces the previous sub-agent pattern with direct tool access.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import List, Optional

import httpx
from llama_index.core.tools import BaseTool

from config import get_config

logger = logging.getLogger("uvicorn.error")

# Cached tools (loaded once per process, None means not attempted yet)
_mcp_tools: Optional[List[BaseTool]] = None
_mcp_load_attempted: bool = False
# Keep MCP session alive for tool execution
_mcp_session_context = None


async def _check_wizard_backend_health(api_url: str, timeout: float = 5.0) -> bool:
    """
    Check if the wizard backend is reachable.

    Args:
        api_url: The wizard backend URL
        timeout: Connection timeout in seconds

    Returns:
        True if backend is reachable, False otherwise
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{api_url}/api/health")
            return response.status_code == 200
    except httpx.ConnectError:
        logger.warning(f"Wizard backend not reachable at {api_url}")
        return False
    except httpx.TimeoutException:
        logger.warning(f"Wizard backend timeout at {api_url}")
        return False
    except Exception as e:
        logger.warning(f"Wizard backend health check failed: {e}")
        return False


async def create_wizard_tools(force_reload: bool = False) -> List[BaseTool]:
    """
    Create wizard tools by connecting to MCP server.

    The MCP server provides 50+ tools for interacting with the Allora wizard backend.

    If the wizard backend is not reachable, returns an empty list but does NOT
    cache the failure, allowing retry on subsequent calls.

    Environment:
        MCP_WIZARD_PACKAGE: npm package name (e.g., "allora-wizard-mcp")
                           Must be installed globally or npm-linked
        WIZARD_API_URL: URL of the wizard backend

    Args:
        force_reload: If True, reload tools even if already cached

    Returns:
        List of BaseTool instances from the MCP server
    """
    global _mcp_tools, _mcp_load_attempted

    # Return cached tools if available and not forcing reload
    if _mcp_tools is not None and not force_reload:
        return _mcp_tools

    config = get_config()

    # Check if MCP is configured
    if not config.wizard.mcp_package:
        logger.info("MCP_WIZARD_PACKAGE not set. Wizard tools unavailable.")
        return []

    # Check if wizard backend is reachable before loading MCP tools
    if not await _check_wizard_backend_health(config.wizard.api_url):
        logger.warning(
            f"Wizard backend not available at {config.wizard.api_url}. "
            "Wizard tools will be unavailable for this request. "
            "Will retry on next request."
        )
        # Don't cache failure - allow retry on next request
        return []

    try:
        from mcp import StdioServerParameters
        from mcp.client.stdio import stdio_client
        from mcp.client.session import ClientSession
        from llama_index.tools.mcp import aget_tools_from_mcp_url

        global _mcp_session_context

        package = config.wizard.mcp_package
        logger.info(f"Loading MCP tools from: {package}")
        logger.info(f"MCP server will connect to wizard backend at: {config.wizard.api_url}")

        # Create MCP server parameters with explicit environment
        # This ensures the subprocess receives WIZARD_API_URL
        env = {**os.environ, "WIZARD_API_URL": config.wizard.api_url}
        server_params = StdioServerParameters(
            command="npx",
            args=[package],
            env=env,
        )

        # Start the MCP server process and create session
        stdio_ctx = stdio_client(server_params)
        read, write = await stdio_ctx.__aenter__()

        session_ctx = ClientSession(read, write)
        session = await session_ctx.__aenter__()
        await session.initialize()

        # Keep contexts alive for later cleanup
        _mcp_session_context = (stdio_ctx, session_ctx, read, write)

        # Get tools using the session with correct environment
        _mcp_tools = await aget_tools_from_mcp_url(package, client=session)
        _mcp_load_attempted = True

        logger.info(f"Loaded {len(_mcp_tools)} wizard tools from MCP server")

        # Log tool names for debugging
        if _mcp_tools:
            tool_names = [t.metadata.name for t in _mcp_tools[:5]]
            logger.debug(f"MCP tools (first 5): {tool_names}...")

        return _mcp_tools

    except ImportError as e:
        logger.warning(f"MCP dependencies not installed: {e}. Run: pip install llama-index-tools-mcp mcp")
        return []
    except Exception as e:
        logger.error(f"Failed to load MCP tools: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # Don't cache failure - allow retry on next request
        return []


async def clear_wizard_tools_cache():
    """Clear the cached wizard tools, forcing reload on next call."""
    global _mcp_tools, _mcp_load_attempted, _mcp_session_context

    # Cleanup existing session if any
    if _mcp_session_context is not None:
        stdio_ctx, session_ctx, _, _ = _mcp_session_context
        try:
            await session_ctx.__aexit__(None, None, None)
            await stdio_ctx.__aexit__(None, None, None)
        except Exception as e:
            logger.warning(f"Error cleaning up MCP session: {e}")
        _mcp_session_context = None

    _mcp_tools = None
    _mcp_load_attempted = False
    logger.info("Wizard tools cache cleared")
