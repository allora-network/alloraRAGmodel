"""
Slack request types and parsing functions
"""

import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
from exceptions import SlackIntegrationError

logger = logging.getLogger("uvicorn.error")


@dataclass
class SlackRequest:
    """Parsed Slack request data"""
    # Request metadata
    request_id: str
    request_type: str  # "url_verification", "event_callback", etc.
    
    # Event data (for event_callback requests)
    event_type: Optional[str] = None
    channel: Optional[str] = None
    user: Optional[str] = None
    text: Optional[str] = None
    thread_ts: Optional[str] = None
    message_ts: Optional[str] = None
    
    # URL verification data
    challenge: Optional[str] = None
    
    # Bot detection
    is_bot_message: bool = False
    
    # Message type detection
    is_dm: bool = False
    is_mention: bool = False
    
    # Session ID for agent
    session_id: Optional[str] = None


def parse_slack_request(json_body: Dict[str, Any], request_id: str) -> Tuple[SlackRequest, bool]:
    """Parse Slack request and determine if we should process it
    
    Args:
        json_body: Raw JSON body from Slack
        request_id: Request ID for logging
        
    Returns:
        Tuple of (SlackRequest, should_process)
        should_process is True if this is a message we want to handle
        
    Raises:
        SlackIntegrationError: If the request is malformed or missing required fields
    """
    try:
        request_type = json_body.get("type")
        if not request_type:
            raise SlackIntegrationError("Missing 'type' field in Slack request")
        
        slack_request = SlackRequest(
            request_id=request_id,
            request_type=request_type
        )
        
        # Handle URL verification challenge
        if request_type == "url_verification":
            challenge = json_body.get("challenge")
            if not challenge:
                raise SlackIntegrationError("Missing 'challenge' field in URL verification request")
            
            slack_request.challenge = challenge
            logger.info(f"[{request_id}] URL verification challenge received")
            return slack_request, True
        
        # Handle event callbacks
        if request_type == "event_callback":
            event_data = json_body.get("event", {})
            if not event_data:
                raise SlackIntegrationError("Missing 'event' field in event_callback request")
            
            event_type = event_data.get("type")
            if not event_type:
                raise SlackIntegrationError("Missing 'type' field in event data")
            
            slack_request.event_type = event_type
            slack_request.channel = event_data.get("channel")
            slack_request.user = event_data.get("user")
            slack_request.text = event_data.get("text", "")
            slack_request.thread_ts = event_data.get("thread_ts")
            slack_request.message_ts = event_data.get("ts")
            
            # Check if this is a bot message (ignore these to prevent loops)
            slack_request.is_bot_message = (
                event_data.get("bot_id") is not None or 
                event_data.get("subtype") == "bot_message"
            )
            
            if slack_request.is_bot_message:
                logger.info(f"[{request_id}] Ignoring bot message")
                return slack_request, False
            
            # Check if this is a message type we handle
            if event_type not in ["message", "app_mention"]:
                logger.info(f"[{request_id}] Unhandled event type: {event_type}")
                return slack_request, False
            
            # Validate required fields for message processing
            if not slack_request.channel:
                raise SlackIntegrationError("Missing 'channel' field in message event")
            
            if not slack_request.message_ts:
                raise SlackIntegrationError("Missing 'ts' field in message event")
            
            # Determine message type
            slack_request.is_dm = slack_request.channel.startswith("D")
            slack_request.is_mention = event_type == "app_mention"
            
            # Generate session ID for agent
            slack_request.session_id = f"{slack_request.channel}.{slack_request.message_ts}"
            
            logger.info(f"[{request_id}] Processing {event_type} from user {slack_request.user} in channel {slack_request.channel}")
            return slack_request, True
        
        # Unhandled request type
        logger.warning(f"[{request_id}] Unhandled request type: {request_type}")
        return slack_request, False
        
    except SlackIntegrationError:
        # Re-raise Slack integration errors
        raise
    except Exception as e:
        # Convert other errors to SlackIntegrationError
        raise SlackIntegrationError(f"Error parsing Slack request: {str(e)}")