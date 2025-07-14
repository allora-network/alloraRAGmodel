import logging
import asyncio
import time
import os
import re
from typing import Dict, Any, Optional, List
import httpx
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from utils import pretty_print

logger = logging.getLogger("uvicorn.error")

async def process_slack_message(event: Dict[str, Any], request_id: str, agent):
    """Process Slack messages asynchronously"""
    
    try:
        channel = event.get("channel")
        if channel is None:
            raise NoChannelSpecified()

        user = event.get("user")
        text = event.get("text", "")
        thread_ts = event.get("thread_ts")
        message_ts = event.get("ts")
        event_type = event.get("type")
        
        logger.info(f"[{request_id}] Processing {event_type} from user {user} in channel {channel}")
        
        # Determine if this is a DM or channel mention
        is_dm = channel.startswith('D')
        is_mention = event_type == "app_mention"
        
        # Extract clean message text
        clean_text = extract_message_text(text, is_mention)
        
        if not clean_text.strip():
            logger.info(f"[{request_id}] Empty message after cleaning, skipping")
            return
        
        # Get response from RAG system
        logger.info(f"[{request_id}] Querying RAG system: '{clean_text[:50]}{'...' if len(clean_text) > 50 else ''}'")
        
        t0 = time.time()
        answer, sources, image_paths = await agent.answer_allora_query(request_id, clean_text)
        query_time = time.time() - t0

        logger.info(f"[{request_id}] Raw answer type: {type(answer)}")
        logger.info(f"[{request_id}] Raw answer value: '{answer}'")
        logger.info(f"[{request_id}] Raw answer repr: {repr(answer)}")
        
        pretty_print({
            "answer": answer,
            "sources": sources,
        })
        
        # Format response for Slack
        formatted_response = format_slack_response(answer, sources, is_dm)
        
        # Send response with optional images
        await send_slack_response(
            channel=channel,
            text=formatted_response,
            thread_ts=thread_ts if not is_dm else None,
            message_ts=message_ts,
            image_paths=image_paths,
        )
        
        logger.info(f"[{request_id}] Response sent - Query time: {query_time:.2f}s")

    except NoChannelSpecified as e:
        logger.error("no channel specified in slack event")
        
    except Exception as e:
        logger.error(f"[{request_id}] Error processing message: {str(e)}")
        
        # Send error response to user
        try:
            await send_slack_response(
                channel=event["channel"],
                text=f"Sorry, I encountered an error processing your request. Please try again later.  ```{str(e)}```",
                thread_ts=event.get("thread_ts") if not event.get("channel", "").startswith('D') else None,
                message_ts=event.get("ts")
            )
        except Exception as send_error:
            logger.error(f"[{request_id}] Failed to send error response: {str(send_error)}")


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


def format_slack_response(answer: str, sources: List[str], is_dm: bool) -> str:
    """Format the RAG response for Slack with proper markdown"""
    
    # Format the main response
    formatted_response = answer
    
    # Add sources if available
    if sources:
        source_text = "\n\n*Sources:*\n"
        for i, source in enumerate(sources[:3], 1):  # Limit to 3 sources for readability
            # Clean up source paths for better readability
            clean_source = source.split('/')[-1] if '/' in source else source
            source_text += f"• {clean_source}\n"
        
        formatted_response += source_text
    
    # Add bot signature for DMs
    if is_dm:
        formatted_response += "\n\n_Powered by Allie - Allora Labs Assistant_"
    
    return formatted_response


async def upload_file_to_slack(bot_token: str, channel: str, file_path: str, title: str, thread_ts: Optional[str] = None):
    """Upload a local file to Slack"""
    logger.info(f"Uploading file to Slack: {file_path}")
    
    try:
        client = WebClient(token=bot_token)
        
        upload_params = {
            "channel": channel,
            "file": file_path,
            "title": title,
            "filename": os.path.basename(file_path)
        }
        
        if thread_ts:
            upload_params["thread_ts"] = thread_ts
        
        response = client.files_upload_v2(**upload_params)
        
        if response.get("ok"):
            file_info = response.get("file", {})
            file_id = file_info.get("id", "unknown")
            logger.info(f"Successfully uploaded file to Slack: {file_id}")
        else:
            error = response.get("error", "Unknown error")
            logger.error(f"Slack file upload failed: {error}")
            raise Exception(f"Slack file upload error: {error}")
            
    except SlackApiError as e:
        error_msg = e.response.get('error', 'Unknown error')
        logger.error(f"Slack API error during file upload: {error_msg}")
        raise Exception(f"Slack API error: {error_msg}")
    except Exception as e:
        logger.error(f"Error uploading file to Slack: {str(e)}")
        raise


async def send_slack_response(channel: str, text: str, thread_ts: Optional[str] = None, message_ts: Optional[str] = None, image_paths: list[str] = []):
    """Send response to Slack using Web API"""
    
    bot_token = os.getenv("SLACK_BOT_TOKEN")
    if not bot_token:
        logger.error("SLACK_BOT_TOKEN environment variable not set")
        raise ValueError("Slack bot token not configured")
    
    # Prepare the message payload
    payload = {
        "channel": channel,
        "text": text,  # Fallback text for notifications
        "blocks": [],
    }
    
    # Add thread timestamp for threaded responses
    if thread_ts:
        payload["thread_ts"] = thread_ts
    elif message_ts and not channel.startswith('D'):  # Use message_ts as thread_ts for channel responses
        payload["thread_ts"] = message_ts

    payload["blocks"].extend([{
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": text,
        },
    }])
    
    # Handle images - upload local files, embed URLs directly
    for i, image_path in enumerate(image_paths):
        if image_path.startswith('http'):
            # It's a URL (DALL-E image) - add directly to blocks
            payload["blocks"].append({
                "type": "image",
                "image_url": image_path,
                "title": {
                    "type": "plain_text",
                    "text": f"Generated Image {i+1}",
                    "emoji": True,
                },
            })
        elif os.path.exists(image_path):
            # It's a local file (chart) - upload to Slack
            try:
                logger.info(f"Uploading local chart file to Slack: {image_path}")
                await upload_file_to_slack(bot_token, channel, image_path, "Generated Chart", thread_ts or message_ts)
            except Exception as upload_error:
                logger.error(f"Failed to upload chart file to Slack: {upload_error}")
                # Add a text block mentioning the failed upload
                payload["blocks"].append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"📊 _Chart was generated but couldn't be uploaded: {os.path.basename(image_path)}_"
                    }
                })

    print("PAYLOAD")
    pretty_print(payload)

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://slack.com/api/chat.postMessage",
            headers={
                "Authorization": f"Bearer {bot_token}",
                "Content-Type": "application/json"
            },
            json=payload
        )
        
        if response.status_code != 200:
            logger.error(f"Failed to send Slack message: {response.status_code} - {response.text}")
            raise Exception(f"Slack API error: {response.status_code}")
        
        result = response.json()
        if not result.get("ok"):
            error = result.get("error", "Unknown error")
            logger.error(f"Slack API returned error: {error}")
            raise Exception(f"Slack API error: {error}")



# async def upload_image_to_slack(bot_token: str, channel: str, image_path: str, comment: str, thread_ts: Optional[str] = None):
#     """Upload image to Slack using the official Slack SDK"""
#     try:
#         from slack_sdk import WebClient
#         from slack_sdk.errors import SlackApiError
        
#         # Initialize Slack client
#         client = WebClient(token=bot_token)
        
#         filename = os.path.basename(image_path)
#         logger.info(f"Starting Slack upload with SDK: {filename}")
        
#         # Prepare upload parameters
#         upload_params = {
#             "channel": channel,
#             "file": image_path,
#             "title": "Generated Chart",
#             "initial_comment": comment
#         }
        
#         # Add thread timestamp if provided
#         if thread_ts:
#             upload_params["thread_ts"] = thread_ts
        
#         # Upload using the official SDK
#         response = client.files_upload_v2(**upload_params)
        
#         if response.get("ok"):
#             file_info = response.get("file", {})
#             file_id = file_info.get("id", "unknown")
#             logger.info(f"Successfully uploaded chart to Slack: {file_id}")
#         else:
#             error = response.get("error", "Unknown error")
#             logger.error(f"Slack SDK upload failed: {error}")
#             raise Exception(f"Slack SDK upload error: {error}")
            
#     except SlackApiError as e:
#         error_msg = e.response.get('error', 'Unknown error')
#         logger.error(f"Slack API error during upload: {error_msg}")
#         raise Exception(f"Slack API error: {error_msg}")
#     except ImportError:
#         logger.error("slack-sdk not available, falling back to manual upload")
#         # Fallback to the previous implementation if SDK not available
#         await upload_image_to_slack_manual(bot_token, channel, image_path, comment, thread_ts)
#     except Exception as e:
#         logger.error(f"Error uploading image to Slack with SDK: {str(e)}")
#         raise


# async def upload_image_to_slack_manual(bot_token: str, channel: str, image_path: str, comment: str, thread_ts: Optional[str] = None):
#     """Fallback manual upload implementation"""
#     logger.warning("Using manual upload fallback - consider installing slack-sdk")
#     # Keep the previous implementation as fallback
#     # (Implementation omitted for brevity - can restore if needed)


class NoChannelSpecified(Exception):
    pass