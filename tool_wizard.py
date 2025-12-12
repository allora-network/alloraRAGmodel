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


class WizardBackendUnavailableError(Exception):
    """Raised when the wizard backend is not reachable."""
    pass


# Cached tools (loaded once per process, None means not attempted yet)
_mcp_tools: Optional[List[BaseTool]] = None
_mcp_load_attempted: bool = False
# Keep MCP session alive for tool execution
_mcp_session_context = None
# Dynamic URL override (takes precedence over config when set)
_wizard_url_override: Optional[str] = None


def get_wizard_url() -> str:
    """Get the current wizard backend URL (override or config)."""
    if _wizard_url_override is not None:
        return _wizard_url_override
    return get_config().wizard.api_url


async def set_wizard_url(url: str) -> None:
    """
    Update the wizard backend URL dynamically without restart.

    Clears the cached tools so the next request will reconnect
    to the new backend.

    Args:
        url: New wizard backend URL
    """
    global _wizard_url_override
    _wizard_url_override = url
    logger.info(f"Wizard URL updated to: {url}")

    # Clear cached tools so next request reconnects to new URL
    await clear_wizard_tools_cache()


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

    Health check runs on every request:
    - If backend is healthy and tools are cached, return cached tools
    - If backend is healthy and tools are not cached, load them
    - If backend is unhealthy and tools are cached, clear cache and return empty
    - If backend is unhealthy and tools are not cached, return empty

    Environment:
        MCP_WIZARD_PACKAGE: Either:
            - Scoped npm package (e.g., "@allora-network/wizard-mcp") - uses npx
            - Binary name (e.g., "allora-wizard-mcp") - runs directly
            For production Docker, use the binary name (installed globally during build)
        WIZARD_API_URL: URL of the wizard backend

    Args:
        force_reload: If True, reload tools even if already cached

    Returns:
        List of BaseTool instances from the MCP server
    """
    global _mcp_tools, _mcp_load_attempted

    config = get_config()

    # Check if MCP is configured
    if not config.wizard.mcp_package:
        logger.info("MCP_WIZARD_PACKAGE not set. Wizard tools unavailable.")
        return []

    # Get current wizard URL (may be dynamically overridden)
    wizard_url = get_wizard_url()

    # Always check if wizard backend is reachable (on every request)
    backend_healthy = await _check_wizard_backend_health(wizard_url)

    if not backend_healthy:
        # Backend is down - clear cache if we have cached tools
        if _mcp_tools is not None:
            logger.warning(
                f"Wizard backend became unavailable at {wizard_url}. "
                "Clearing cached tools."
            )
            await clear_wizard_tools_cache()
        else:
            logger.warning(
                f"Wizard backend not available at {wizard_url}. "
                "Wizard tools will be unavailable for this request."
            )
        # Raise exception so caller can handle appropriately
        raise WizardBackendUnavailableError(
            f"Wizard backend not available at {wizard_url}"
        )

    # Backend is healthy - return cached tools if available and not forcing reload
    if _mcp_tools is not None and not force_reload:
        return _mcp_tools

    try:
        from mcp import StdioServerParameters
        from mcp.client.stdio import stdio_client
        from mcp.client.session import ClientSession
        from llama_index.tools.mcp import aget_tools_from_mcp_url

        global _mcp_session_context

        package = config.wizard.mcp_package
        logger.info(f"Loading MCP tools from: {package}")
        logger.info(f"MCP server will connect to wizard backend at: {wizard_url}")

        # Create MCP server parameters with explicit environment
        # This ensures the subprocess receives WIZARD_API_URL
        env = {**os.environ, "WIZARD_API_URL": wizard_url}

        # Determine how to run the MCP server:
        # - If package starts with "@" (scoped npm package), use npx
        # - Otherwise, assume it's a globally installed binary and run directly
        if package.startswith("@"):
            # Scoped package like @allora-network/wizard-mcp - use npx
            command = "npx"
            args = [package]
            logger.info(f"Using npx to run scoped package: {package}")
        else:
            # Binary name like allora-wizard-mcp - run directly
            command = package
            args = []
            logger.info(f"Running MCP server binary directly: {package}")

        server_params = StdioServerParameters(
            command=command,
            args=args,
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
    """Clear the cached wizard tools, forcing reload on next call.

    Note: MCP sessions use anyio cancel scopes which cannot be exited from
    a different task than they were entered in. We simply abandon the session
    and let the subprocess terminate naturally - any cleanup attempt from a
    different task will cause RuntimeError. The subprocess will exit when it
    detects the broken pipe or when the process is garbage collected.
    """
    global _mcp_tools, _mcp_load_attempted, _mcp_session_context

    # Simply abandon the session without attempting cleanup
    # Attempting to close streams or call __aexit__ from a different task
    # will cause "Attempted to exit cancel scope in a different task" error
    if _mcp_session_context is not None:
        logger.debug("Abandoning MCP session (will be cleaned up by subprocess)")
        _mcp_session_context = None

    _mcp_tools = None
    _mcp_load_attempted = False
    logger.info("Wizard tools cache cleared")
