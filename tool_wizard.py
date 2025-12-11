"""
Allora Topic Wizard tools for LlamaIndex FunctionAgent
Provides read-only query tools for Allora topics, whitelists, scores, and OSM data.
"""

import logging
import json
from typing import Annotated, List, Literal, Optional
import httpx
from llama_index.core.tools import FunctionTool, BaseTool
from config import get_config

logger = logging.getLogger("uvicorn.error")


SERVER_UNAVAILABLE_MESSAGE = (
    "The Allora Topic Wizard service is currently unavailable. "
    "This tool requires the wizard backend to be running. "
    "Please try again later or contact the team if the issue persists."
)


async def _make_request(path: str, params: Optional[dict] = None) -> str:
    """Make HTTP GET request to wizard API and return formatted response."""
    config = get_config()
    url = f"{config.wizard.api_url}{path}"

    # Build headers with optional API key authentication
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
        logger.error(f"Connection failed to wizard API: {url} - server may be down")
        return SERVER_UNAVAILABLE_MESSAGE
    except httpx.ConnectTimeout:
        logger.error(f"Connection timeout to wizard API: {url} - server may be down")
        return SERVER_UNAVAILABLE_MESSAGE
    except httpx.TimeoutException:
        logger.error(f"Request timeout calling wizard API: {url}")
        return "Error: Request timed out. The wizard backend may be slow. Please try again."
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error calling wizard API: {e}")
        if e.response.status_code == 401:
            return "Error: Authentication failed. The WIZARD_API_KEY may be invalid or missing."
        if e.response.status_code == 403:
            return "Error: Access forbidden. You don't have permission to access this resource."
        if e.response.status_code >= 500:
            return SERVER_UNAVAILABLE_MESSAGE
        return f"Error: HTTP {e.response.status_code} - {e.response.text}"
    except Exception as e:
        logger.error(f"Error calling wizard API: {e}")
        # Check for common connection-related error messages
        error_str = str(e).lower()
        if any(x in error_str for x in ["connection", "refused", "unreachable", "resolve"]):
            return SERVER_UNAVAILABLE_MESSAGE
        return f"Error: {str(e)}"


# =============================================================================
# Topic Query Tools
# =============================================================================

async def get_topic(
    topic_id: Annotated[str, "The topic ID to query"],
    network: Annotated[Literal["testnet", "mainnet"], "Network to query (default: testnet)"] = "testnet"
) -> str:
    """Get detailed information about an Allora topic."""
    return await _make_request(f"/api/sdk/query/topic/{topic_id}", {"network": network})


async def get_next_topic_id(
    network: Annotated[Literal["testnet", "mainnet"], "Network to query (default: testnet)"] = "testnet"
) -> str:
    """Get the next available topic ID that will be assigned when a new topic is created."""
    return await _make_request("/api/sdk/query/next-topic-id", {"network": network})


async def is_topic_active(
    topic_id: Annotated[str, "The topic ID to check"],
    network: Annotated[Literal["testnet", "mainnet"], "Network to query (default: testnet)"] = "testnet"
) -> str:
    """Check if a topic is currently active and accepting submissions."""
    return await _make_request(f"/api/sdk/query/topic/{topic_id}/active", {"network": network})


async def get_topic_stake(
    topic_id: Annotated[str, "The topic ID to query"],
    network: Annotated[Literal["testnet", "mainnet"], "Network to query (default: testnet)"] = "testnet"
) -> str:
    """Get the total stake amount in a topic."""
    return await _make_request(f"/api/sdk/query/topic/{topic_id}/stake", {"network": network})


# =============================================================================
# Whitelist Tools
# =============================================================================

async def is_worker_whitelist_enabled(
    topic_id: Annotated[str, "The topic ID to check"],
    network: Annotated[Literal["testnet", "mainnet"], "Network to query (default: testnet)"] = "testnet"
) -> str:
    """Check if worker whitelisting is enabled for a topic."""
    return await _make_request(f"/api/sdk/query/topic/{topic_id}/worker-whitelist-enabled", {"network": network})


async def is_reputer_whitelist_enabled(
    topic_id: Annotated[str, "The topic ID to check"],
    network: Annotated[Literal["testnet", "mainnet"], "Network to query (default: testnet)"] = "testnet"
) -> str:
    """Check if reputer whitelisting is enabled for a topic."""
    return await _make_request(f"/api/sdk/query/topic/{topic_id}/reputer-whitelist-enabled", {"network": network})


async def is_worker_whitelisted(
    topic_id: Annotated[str, "The topic ID to check"],
    address: Annotated[str, "Worker address to check (allo1...)"],
    network: Annotated[Literal["testnet", "mainnet"], "Network to query (default: testnet)"] = "testnet"
) -> str:
    """Check if a specific worker address is whitelisted for a topic."""
    return await _make_request(f"/api/sdk/query/topic/{topic_id}/worker/{address}/whitelisted", {"network": network})


async def is_reputer_whitelisted(
    topic_id: Annotated[str, "The topic ID to check"],
    address: Annotated[str, "Reputer address to check (allo1...)"],
    network: Annotated[Literal["testnet", "mainnet"], "Network to query (default: testnet)"] = "testnet"
) -> str:
    """Check if a specific reputer address is whitelisted for a topic."""
    return await _make_request(f"/api/sdk/query/topic/{topic_id}/reputer/{address}/whitelisted", {"network": network})


# =============================================================================
# Registration Tools
# =============================================================================

async def is_worker_registered(
    topic_id: Annotated[str, "The topic ID to check"],
    address: Annotated[str, "Worker address to check (allo1...)"],
    network: Annotated[Literal["testnet", "mainnet"], "Network to query (default: testnet)"] = "testnet"
) -> str:
    """Check if a worker is registered for a topic."""
    return await _make_request(f"/api/sdk/query/topic/{topic_id}/worker/{address}/registered", {"network": network})


async def is_reputer_registered(
    topic_id: Annotated[str, "The topic ID to check"],
    address: Annotated[str, "Reputer address to check (allo1...)"],
    network: Annotated[Literal["testnet", "mainnet"], "Network to query (default: testnet)"] = "testnet"
) -> str:
    """Check if a reputer is registered for a topic."""
    return await _make_request(f"/api/sdk/query/topic/{topic_id}/reputer/{address}/registered", {"network": network})


# =============================================================================
# Inference Tools
# =============================================================================

async def get_latest_inferences(
    topic_id: Annotated[str, "The topic ID to query"],
    network: Annotated[Literal["testnet", "mainnet"], "Network to query (default: testnet)"] = "testnet"
) -> str:
    """Get the latest inferences submitted to a topic."""
    return await _make_request(f"/api/sdk/query/topic/{topic_id}/inferences", {"network": network})


async def get_network_inference(
    topic_id: Annotated[str, "The topic ID to query"],
    network: Annotated[Literal["testnet", "mainnet"], "Network to query (default: testnet)"] = "testnet"
) -> str:
    """Get the aggregated network inference for a topic."""
    return await _make_request(f"/api/sdk/query/topic/{topic_id}/network-inference", {"network": network})


# =============================================================================
# Score Tools
# =============================================================================

async def get_inferer_score(
    topic_id: Annotated[str, "The topic ID to query"],
    address: Annotated[str, "Inferer address (allo1...)"],
    network: Annotated[Literal["testnet", "mainnet"], "Network to query (default: testnet)"] = "testnet"
) -> str:
    """Get the score for a specific inferer (worker submitting inferences) in a topic."""
    return await _make_request(f"/api/sdk/query/topic/{topic_id}/inferer/{address}/score", {"network": network})


async def get_forecaster_score(
    topic_id: Annotated[str, "The topic ID to query"],
    address: Annotated[str, "Forecaster address (allo1...)"],
    network: Annotated[Literal["testnet", "mainnet"], "Network to query (default: testnet)"] = "testnet"
) -> str:
    """Get the score for a specific forecaster in a topic."""
    return await _make_request(f"/api/sdk/query/topic/{topic_id}/forecaster/{address}/score", {"network": network})


async def get_reputer_score(
    topic_id: Annotated[str, "The topic ID to query"],
    address: Annotated[str, "Reputer address (allo1...)"],
    network: Annotated[Literal["testnet", "mainnet"], "Network to query (default: testnet)"] = "testnet"
) -> str:
    """Get the score for a specific reputer in a topic."""
    return await _make_request(f"/api/sdk/query/topic/{topic_id}/reputer/{address}/score", {"network": network})


async def get_reputer_stake_in_topic(
    topic_id: Annotated[str, "The topic ID to query"],
    address: Annotated[str, "Reputer address (allo1...)"],
    network: Annotated[Literal["testnet", "mainnet"], "Network to query (default: testnet)"] = "testnet"
) -> str:
    """Get the stake amount for a specific reputer in a topic."""
    return await _make_request(f"/api/sdk/query/topic/{topic_id}/reputer/{address}/stake", {"network": network})


# =============================================================================
# Service Tools
# =============================================================================

async def list_services(
    service_type: Annotated[Literal["reputer", "forecaster"], "Type of service to list"],
    network: Annotated[Optional[Literal["testnet", "mainnet"]], "Filter by network"] = None,
    customer: Annotated[Optional[str], "Filter by customer ID (e.g., '000000000')"] = None,
    service: Annotated[Optional[str], "Filter by service type (e.g., 'returns-prediction', 'price-prediction')"] = None,
    instance: Annotated[Optional[str], "Filter by instance number (e.g., '1', '2')"] = None,
    topic: Annotated[Optional[str], "Filter by topic ID (forecasters only)"] = None
) -> str:
    """List available reputer or forecaster services with optional filters."""
    endpoint = "/api/wizard/list-forecasters" if service_type == "forecaster" else "/api/wizard/list-reputers"

    params = {}
    if network:
        params["network"] = network
    if customer:
        params["customer"] = customer
    if service:
        params["service"] = service
    if instance:
        params["instance"] = instance
    if topic:
        params["topic"] = topic

    return await _make_request(endpoint, params if params else None)


async def list_competitions() -> str:
    """List existing Forge competitions."""
    return await _make_request("/api/wizard/list-competitions")


# =============================================================================
# OSM Tools
# =============================================================================

async def get_osm_config(
    service_type: Annotated[Literal["reputer", "forecaster"], "Type of service to query"],
    network: Annotated[Literal["testnet", "mainnet"], "Network to query (default: testnet)"] = "testnet",
    customer: Annotated[Optional[str], "Filter by customer ID (e.g., '000000000')"] = None,
    service: Annotated[Optional[str], "Filter by service name (e.g., 'returns-prediction', 'price-prediction')"] = None,
    instance: Annotated[Optional[str], "Filter by instance number (e.g., '1', '2')"] = None,
    topic_id: Annotated[Optional[str], "Filter by topic ID"] = None
) -> str:
    """Query detailed OSM configurations including topics served, pairs, timeframes, loss methods, ground truth providers, and stake amounts."""
    params = {
        "network": network,
        "purpose": service_type,
        "detailed": "true"
    }
    if customer:
        params["customer"] = customer
    if service:
        params["service"] = service
    if instance:
        params["instance"] = instance
    if topic_id:
        params["topic"] = topic_id

    return await _make_request("/api/osm/config", params)


async def get_osm_wallets(
    network: Annotated[Literal["testnet", "mainnet"], "Network to query (default: testnet)"] = "testnet",
    customer: Annotated[Optional[str], "Filter by customer ID"] = None,
    service: Annotated[Optional[str], "Filter by service name"] = None,
    instance: Annotated[Optional[str], "Filter by instance number"] = None,
    topic_id: Annotated[Optional[str], "Filter by topic ID"] = None,
    purpose: Annotated[Optional[Literal["reputer", "forecaster", "worker"]], "Filter by wallet purpose"] = None,
    deployed_only: Annotated[Optional[bool], "Only return wallets for services actually deployed in Kubernetes"] = None
) -> str:
    """Query wallet addresses from OSM for forecasters, reputers, or workers."""
    params = {"network": network}
    if customer:
        params["customer"] = customer
    if service:
        params["service"] = service
    if instance:
        params["instance"] = instance
    if topic_id:
        params["topic"] = topic_id
    if purpose:
        params["purpose"] = purpose
    if deployed_only:
        params["deployedOnly"] = "true"

    return await _make_request("/api/osm/wallets", params)


async def get_osm_topic_details(
    topic_id: Annotated[str, "The topic ID to query"],
    network: Annotated[Literal["testnet", "mainnet"], "Network to query (default: testnet)"] = "testnet"
) -> str:
    """Get comprehensive OSM configuration for a specific topic, showing all reputers and forecasters serving it with their full configurations."""
    return await _make_request(f"/api/osm/topic/{topic_id}", {"network": network})


# =============================================================================
# Kubernetes Tools
# =============================================================================

async def get_k8s_deployments(
    network: Annotated[Literal["testnet", "mainnet"], "Network environment (default: testnet)"] = "testnet",
    namespace: Annotated[Optional[str], "Kubernetes namespace. If not specified with filters, searches all namespaces."] = None,
    topic_id: Annotated[Optional[str], "Filter by topic ID label"] = None,
    service_type: Annotated[Optional[Literal["forecaster", "worker", "reputer", "all"]], "Type of service to query (default: all)"] = "all",
    label_selector: Annotated[Optional[str], "Label selector (e.g., 'app=forecaster,topic=123')"] = None
) -> str:
    """Inspect Kubernetes deployments for Allora forecasters, workers, and reputers. Shows replica status, images, and labels."""
    params = {"network": network}
    if namespace:
        params["namespace"] = namespace
    elif topic_id or label_selector:
        params["namespace"] = "all"
    else:
        params["namespace"] = "default"

    if topic_id:
        params["topicId"] = topic_id
    if service_type and service_type != "all":
        params["serviceType"] = service_type
    if label_selector:
        params["labelSelector"] = label_selector

    return await _make_request("/api/k8s/deployments", params)


async def get_k8s_configmaps(
    network: Annotated[Literal["testnet", "mainnet"], "Network environment (default: testnet)"] = "testnet",
    namespace: Annotated[Optional[str], "Kubernetes namespace. If not specified with filters, searches all namespaces."] = None,
    name: Annotated[Optional[str], "Specific ConfigMap name to fetch"] = None,
    topic_id: Annotated[Optional[str], "Filter by topic ID label"] = None,
    service_type: Annotated[Optional[Literal["forecaster", "worker", "reputer"]], "Filter by service type"] = None,
    include_data: Annotated[bool, "Include ConfigMap data contents (default: true)"] = True,
    label_selector: Annotated[Optional[str], "Label selector (e.g., 'workload.allora.network/purpose=forecaster')"] = None
) -> str:
    """Inspect Kubernetes ConfigMaps containing Allora service configurations. Shows topic settings, pairs, timeframes, and other configuration data."""
    params = {
        "network": network,
        "includeData": str(include_data).lower()
    }
    if namespace:
        params["namespace"] = namespace
    elif topic_id or label_selector or service_type:
        params["namespace"] = "all"
    else:
        params["namespace"] = "default"

    if name:
        params["name"] = name
    if topic_id:
        params["topicId"] = topic_id
    if service_type:
        params["serviceType"] = service_type
    if label_selector:
        params["labelSelector"] = label_selector

    return await _make_request("/api/k8s/configmaps", params)


async def get_k8s_pod_status(
    network: Annotated[Literal["testnet", "mainnet"], "Network environment (default: testnet)"] = "testnet",
    namespace: Annotated[Optional[str], "Kubernetes namespace. If not specified with filters, searches all namespaces."] = None,
    topic_id: Annotated[Optional[str], "Filter by topic ID label"] = None,
    service_type: Annotated[Optional[Literal["forecaster", "worker", "reputer"]], "Filter by service type"] = None,
    deployment_name: Annotated[Optional[str], "Filter by deployment name"] = None,
    include_recent_logs: Annotated[bool, "Include recent log lines (default: false)"] = False,
    label_selector: Annotated[Optional[str], "Label selector (e.g., 'app=forecaster,topic=123')"] = None
) -> str:
    """Check Kubernetes pod status for Allora services. Shows pod health, restart counts, container states, and optionally recent logs."""
    params = {
        "network": network,
        "includeRecentLogs": str(include_recent_logs).lower()
    }
    if namespace:
        params["namespace"] = namespace
    elif topic_id or label_selector or service_type:
        params["namespace"] = "all"
    else:
        params["namespace"] = "default"

    if topic_id:
        params["topicId"] = topic_id
    if service_type:
        params["serviceType"] = service_type
    if deployment_name:
        params["deploymentName"] = deployment_name
    if label_selector:
        params["labelSelector"] = label_selector

    return await _make_request("/api/k8s/pods", params)


# =============================================================================
# Tool Creation
# =============================================================================

def create_wizard_tools() -> List[BaseTool]:
    """Create all wizard tools for registration with the agent."""
    tools = [
        # Topic Query Tools
        FunctionTool.from_defaults(
            fn=get_topic,
            name="get_topic",
            description="Get detailed information about an Allora topic including metadata, loss method, epoch settings, and configuration."
        ),
        FunctionTool.from_defaults(
            fn=get_next_topic_id,
            name="get_next_topic_id",
            description="Get the next available topic ID that will be assigned when a new topic is created."
        ),
        FunctionTool.from_defaults(
            fn=is_topic_active,
            name="is_topic_active",
            description="Check if a topic is currently active and accepting submissions."
        ),
        FunctionTool.from_defaults(
            fn=get_topic_stake,
            name="get_topic_stake",
            description="Get the total stake amount in a topic."
        ),

        # Whitelist Tools
        FunctionTool.from_defaults(
            fn=is_worker_whitelist_enabled,
            name="is_worker_whitelist_enabled",
            description="Check if worker whitelisting is enabled for a topic."
        ),
        FunctionTool.from_defaults(
            fn=is_reputer_whitelist_enabled,
            name="is_reputer_whitelist_enabled",
            description="Check if reputer whitelisting is enabled for a topic."
        ),
        FunctionTool.from_defaults(
            fn=is_worker_whitelisted,
            name="is_worker_whitelisted",
            description="Check if a specific worker address is whitelisted for a topic."
        ),
        FunctionTool.from_defaults(
            fn=is_reputer_whitelisted,
            name="is_reputer_whitelisted",
            description="Check if a specific reputer address is whitelisted for a topic."
        ),

        # Registration Tools
        FunctionTool.from_defaults(
            fn=is_worker_registered,
            name="is_worker_registered",
            description="Check if a worker is registered for a topic."
        ),
        FunctionTool.from_defaults(
            fn=is_reputer_registered,
            name="is_reputer_registered",
            description="Check if a reputer is registered for a topic."
        ),

        # Inference Tools
        FunctionTool.from_defaults(
            fn=get_latest_inferences,
            name="get_latest_inferences",
            description="Get the latest inferences submitted to a topic."
        ),
        FunctionTool.from_defaults(
            fn=get_network_inference,
            name="get_network_inference",
            description="Get the aggregated network inference for a topic."
        ),

        # Score Tools
        FunctionTool.from_defaults(
            fn=get_inferer_score,
            name="get_inferer_score",
            description="Get the score for a specific inferer (worker submitting inferences) in a topic."
        ),
        FunctionTool.from_defaults(
            fn=get_forecaster_score,
            name="get_forecaster_score",
            description="Get the score for a specific forecaster in a topic."
        ),
        FunctionTool.from_defaults(
            fn=get_reputer_score,
            name="get_reputer_score",
            description="Get the score for a specific reputer in a topic."
        ),
        FunctionTool.from_defaults(
            fn=get_reputer_stake_in_topic,
            name="get_reputer_stake_in_topic",
            description="Get the stake amount for a specific reputer in a topic."
        ),

        # Service Tools
        FunctionTool.from_defaults(
            fn=list_services,
            name="list_services",
            description="List available reputer or forecaster services with optional filters. Use this to discover what services are deployed."
        ),
        FunctionTool.from_defaults(
            fn=list_competitions,
            name="list_competitions",
            description="List existing Forge competitions."
        ),

        # OSM Tools
        FunctionTool.from_defaults(
            fn=get_osm_config,
            name="get_osm_config",
            description="Query detailed OSM configurations including topics served, pairs, timeframes, loss methods, ground truth providers, and stake amounts."
        ),
        FunctionTool.from_defaults(
            fn=get_osm_wallets,
            name="get_osm_wallets",
            description="Query wallet addresses from OSM for forecasters, reputers, or workers. Useful for checking addresses before whitelisting."
        ),
        FunctionTool.from_defaults(
            fn=get_osm_topic_details,
            name="get_osm_topic_details",
            description="Get comprehensive OSM configuration for a specific topic, showing all reputers and forecasters serving it with their full configurations."
        ),

        # Kubernetes Tools
        FunctionTool.from_defaults(
            fn=get_k8s_deployments,
            name="get_k8s_deployments",
            description="Inspect Kubernetes deployments for Allora forecasters, workers, and reputers. Shows replica status, images, and labels."
        ),
        FunctionTool.from_defaults(
            fn=get_k8s_configmaps,
            name="get_k8s_configmaps",
            description="Inspect Kubernetes ConfigMaps containing Allora service configurations. Shows topic settings, pairs, timeframes, and other configuration data."
        ),
        FunctionTool.from_defaults(
            fn=get_k8s_pod_status,
            name="get_k8s_pod_status",
            description="Check Kubernetes pod status for Allora services. Shows pod health, restart counts, container states, and optionally recent logs."
        ),
    ]

    logger.info(f"Created {len(tools)} wizard tools")
    return tools
