import logging
import os
import re
from typing import Optional, List
import httpx
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from config import get_config
from slack_types import SlackRequest
from utils import pretty_print

logger = logging.getLogger("uvicorn.error")

async def send_slack_response(slack_request: SlackRequest, answer: str, sources: List[str], image_paths: List[str]):
    """Process Slack messages asynchronously"""
    
    try:
        # Format response for Slack
        formatted_response = format_slack_response(answer, sources, slack_request.is_dm)
        
        # Send response with optional images
        await _send_slack_response(
            channel=slack_request.channel,
            text=formatted_response,
            thread_ts=slack_request.thread_ts if not slack_request.is_dm else None,
            message_ts=slack_request.message_ts,
            image_paths=image_paths,
        )

    except Exception as e:
        logger.error(f"[{slack_request.request_id}] Error processing message: {str(e)}")
        
        # Send error response to user
        try:
            await _send_slack_response(
                channel=slack_request.channel,
                text=f"Sorry, I encountered an error processing your request. Please try again later.  ```{str(e)}```",
                thread_ts=slack_request.thread_ts if not slack_request.is_dm else None,
                message_ts=slack_request.message_ts
            )
        except Exception as send_error:
            logger.error(f"[{slack_request.request_id}] Failed to send error response: {str(send_error)}")


def format_slack_response(answer: str, sources: List[str], is_dm: bool) -> str:
    """Format the RAG response for Slack with proper markdown"""
    
    # Format the main response
    formatted_response = answer
    
    # Add sources if available
    if sources:
        source_text = "\n\n*Sources:*\n"
        # config = get_config()
        # for i, source in enumerate(sources[:config.slack.max_sources_displayed], 1):  # Limit sources for readability
        #     # Clean up source paths for better readability
        #     clean_source = source.split('/')[-1] if '/' in source else source
        #     source_text += f"• {clean_source}\n"
        
        formatted_response += source_text
    
    # # Add bot signature for DMs
    # if is_dm:
    #     formatted_response += "\n\n_Powered by Allie - Allora Labs Assistant_"
    
    return formatted_response


async def upload_file_to_slack(channel: str, file_path: str, title: str):
    """Upload a local file to Slack"""
    logger.info(f"Uploading file to Slack: {file_path}")

    try:
        config = get_config()
        client = WebClient(token=config.slack.bot_token)

        with open(file_path, 'rb') as f:
            response = client.files_upload_v2(
                filename=os.path.basename(file_path),
                file=f,
                channel=channel,
            )
        
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


def _split_text_for_slack(text: str, max_length: int) -> List[str]:
    """
    Split text into chunks that fit within Slack's block character limit.
    Tries to split at paragraph boundaries, then sentence boundaries, then word boundaries.
    """
    if len(text) <= max_length:
        return [text]

    chunks = []
    remaining = text

    while remaining:
        if len(remaining) <= max_length:
            chunks.append(remaining)
            break

        # Find a good split point within max_length
        split_point = max_length

        # Try to split at paragraph boundary (double newline)
        last_para = remaining.rfind('\n\n', 0, max_length)
        if last_para > max_length // 2:  # Only use if it's past halfway
            split_point = last_para + 2
        else:
            # Try to split at single newline
            last_newline = remaining.rfind('\n', 0, max_length)
            if last_newline > max_length // 2:
                split_point = last_newline + 1
            else:
                # Try to split at sentence boundary
                for punct in ['. ', '! ', '? ']:
                    last_sent = remaining.rfind(punct, 0, max_length)
                    if last_sent > max_length // 2:
                        split_point = last_sent + 2
                        break
                else:
                    # Fall back to word boundary
                    last_space = remaining.rfind(' ', 0, max_length)
                    if last_space > max_length // 2:
                        split_point = last_space + 1

        chunks.append(remaining[:split_point].rstrip())
        remaining = remaining[split_point:].lstrip()

    return chunks


async def _send_slack_response(channel: str, text: str, thread_ts: Optional[str] = None, message_ts: Optional[str] = None, image_paths: list[str] = []):
    """Send response to Slack using Web API"""
    
    bot_token = os.getenv("SLACK_BOT_TOKEN")
    if not bot_token:
        logger.error("SLACK_BOT_TOKEN environment variable not set")
        raise ValueError("Slack bot token not configured")
    
    # Prepare the message payload
    payload = {
        "channel": channel,
        # "text": text,  # Fallback text for notifications
        "blocks": [],
    }
    
    # Add thread timestamp for threaded responses
    if thread_ts:
        payload["thread_ts"] = thread_ts
    elif message_ts and not channel.startswith("D"):  # Use message_ts as thread_ts for channel responses
        payload["thread_ts"] = message_ts

    # Convert markdown to Slack-compatible format
    text = re.sub(r'```[a-zA-Z0-9_]*\n', '```', text)  # Remove syntax identifiers from code fences
    text = re.sub(r'\*\*(.+?)\*\*', r'*\1*', text)  # Convert **bold** to *bold*
    text = re.sub(r'### (.+)', r'*\1*', text)  # Convert ### headings to *bold*
    text = re.sub(r'## (.+)', r'*\1*', text)  # Convert ## headings to *bold*
    text = re.sub(r'# (.+)', r'*\1*', text)  # Convert # headings to *bold*

    # Slack section blocks have a 3000 char limit - split long messages
    MAX_BLOCK_LENGTH = 2900  # Leave some margin
    text_chunks = _split_text_for_slack(text, MAX_BLOCK_LENGTH)

    for chunk in text_chunks:
        payload["blocks"].append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": chunk,
            },
        })
    
    # Add fallback text for notifications
    payload["text"] = text[:150] + "..." if len(text) > 150 else text
    
    # Handle images - upload local files, embed URLs directly
    for i, image_path in enumerate(image_paths):
        if image_path.startswith('http'):
            # It's a URL (DALL-E or chart image) - add directly to blocks
            payload["blocks"].append({
                "type": "image",
                "image_url": image_path,
                "alt_text": f"Generated Image {i+1}"
            })
            break
        elif os.path.exists(image_path):
            # It's a local file (chart) - upload to Slack
            try:
                logger.info(f"Uploading local chart file to Slack: {image_path}")
                await upload_file_to_slack(channel, image_path, "Generated Chart")
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

    logger.debug(f"Slack message payload: {payload}")

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
            logger.error("Payload:")
            pretty_print(payload)

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


