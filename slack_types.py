"""
Slack request types and parsing functions
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, Any, Literal, Optional, Tuple, Union
from exceptions import SlackIntegrationError

logger = logging.getLogger("uvicorn.error")

@dataclass
class SlackVerificationRequest:
    challenge: str


@dataclass
class SlackRequest:
    # Request metadata
    request_id: str
    request_type: str  # "url_verification", "event_callback", etc.
    
    # Event data (for event_callback requests)
    text: str
    clean_text: str
    channel: str
    event_type: Optional[str] = None
    event_id: Optional[str] = None  # Unique event ID for deduplication
    user: Optional[str] = None
    thread_ts: Optional[str] = None
    message_ts: Optional[str] = None
    
    # Bot detection
    is_bot_message: bool = False
    
    # Message type detection
    is_dm: bool = False
    is_mention: bool = False

    def session_id(self):
        if self.thread_ts:
            # Thread reply: use thread_ts to maintain conversation context
            return f"{self.channel}.{self.thread_ts}"
        else:
            # Top-level message: use message_ts as the thread root
            return f"{self.channel}.{self.message_ts}"



def parse_slack_request(json_body: Dict[str, Any], request_id: str) -> Union[SlackVerificationRequest, SlackRequest, None]:
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
        
        
        # Handle URL verification challenge
        if request_type == "url_verification":
            challenge = json_body.get("challenge")
            if not challenge:
                raise SlackIntegrationError("Missing 'challenge' field in URL verification request")
            
            logger.info(f"[{request_id}] URL verification challenge received")
            return SlackVerificationRequest(challenge=challenge)
        
        # Handle event callbacks
        if request_type == "event_callback":
            event_data = json_body.get("event", {})
            if not event_data:
                raise SlackIntegrationError("Missing 'event' field in event_callback request")
            
            event_type = event_data.get("type")
            if not event_type:
                raise SlackIntegrationError("Missing 'type' field in event data")
            
            text = event_data.get("text", "")
            is_bot_message = (
                event_data.get("bot_id") is not None or
                event_data.get("subtype") == "bot_message"
            )
            channel = event_data.get("channel")
            is_dm = channel.startswith("D")
            is_mention = event_type == "app_mention"
            clean_text = extract_message_text(text, is_mention)

            slack_request = SlackRequest(
                request_id=request_id,
                request_type=request_type,
                event_id=json_body.get("event_id"),
                event_type=event_type,
                channel=channel,
                user=event_data.get("user"),
                text=text,
                clean_text=clean_text,
                is_bot_message=is_bot_message,
                is_dm=is_dm,
                thread_ts=event_data.get("thread_ts"),
                message_ts=event_data.get("ts"),
            )
            
            if slack_request.is_bot_message:
                logger.info(f"[{request_id}] Ignoring bot message")
                return None
            
            # Check if this is a message type we handle
            if event_type not in ["message", "app_mention"]:
                logger.info(f"[{request_id}] Unhandled event type: {event_type}")
                return None
            
            # Validate required fields for message processing
            if not slack_request.channel:
                raise SlackIntegrationError("Missing 'channel' field in message event")
            
            if not slack_request.message_ts:
                raise SlackIntegrationError("Missing 'ts' field in message event")
            
            logger.info(f"[{request_id}] Processing {event_type} from user {slack_request.user} in channel {slack_request.channel}")
            return slack_request
        
        # Unhandled request type
        logger.warning(f"[{request_id}] Unhandled request type: {request_type}")
        return None
        
    except SlackIntegrationError:
        # Re-raise Slack integration errors
        raise
    except Exception as e:
        # Convert other errors to SlackIntegrationError
        raise SlackIntegrationError(f"Error parsing Slack request: {str(e)}")


def extract_message_text(text: str, is_mention: bool) -> str:
    """Extract clean message text, removing bot mentions and formatting"""

    if not text:
        return ""

    # Remove bot mention (format: <@U123456789>)
    if is_mention:
        text = re.sub(r'<@U[A-Z0-9]+>', '', text).strip()

    # Unescape Slack HTML entities
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text
