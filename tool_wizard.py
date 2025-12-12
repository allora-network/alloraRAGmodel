"""
Wizard tools via MCP server integration.

Uses llama-index-tools-mcp to run the MCP server and get tools directly.
This replaces the previous sub-agent pattern with direct tool access.
"""

import logging
from typing import List, Optional

from llama_index.core.tools import BaseTool

from config import get_config

logger = logging.getLogger("uvicorn.error")

# Cached tools (loaded once per process)
_mcp_tools: Optional[List[BaseTool]] = None


async def create_wizard_tools() -> List[BaseTool]:
    """
    Create wizard tools by connecting to MCP server.

    The MCP server provides 50+ tools for interacting with the Allora wizard backend.

    Environment:
        MCP_WIZARD_PACKAGE: npm package name (e.g., "allora-wizard-mcp")
                           Must be installed globally or npm-linked

    Returns:
        List of BaseTool instances from the MCP server
    """
    global _mcp_tools
    if _mcp_tools is not None:
        return _mcp_tools

    config = get_config()

    # Check if MCP is configured
    if not config.wizard.mcp_package:
        logger.info("MCP_WIZARD_PACKAGE not set. Wizard tools unavailable.")
        return []

    try:
        from llama_index.tools.mcp import aget_tools_from_mcp_url

        package = config.wizard.mcp_package
        logger.info(f"Loading MCP tools from: {package}")

        _mcp_tools = await aget_tools_from_mcp_url(package)

        logger.info(f"Loaded {len(_mcp_tools)} wizard tools from MCP server")

        # Log tool names for debugging
        if _mcp_tools:
            tool_names = [t.metadata.name for t in _mcp_tools[:5]]
            logger.debug(f"MCP tools (first 5): {tool_names}...")

        return _mcp_tools

    except ImportError:
        logger.warning("llama-index-tools-mcp not installed. Run: pip install llama-index-tools-mcp")
        return []
    except Exception as e:
        logger.error(f"Failed to load MCP tools: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []
