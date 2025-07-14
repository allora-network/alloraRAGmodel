import logging
import asyncio
import os
from typing import List, Tuple

from llama_cloud import ImageBlock
from llama_index.core.tools import BaseTool
from llama_index.llms.openai import OpenAI, OpenAIResponses
from llama_index.core.agent import FunctionCallingAgent
from llama_index.core.memory import ChatMemoryBuffer

from sysprompt import default_system_prompt
from tool_chart import create_chart_tool
from tool_openai_image import image_tool
from tool_rag import create_rag_tools
from utils import pretty_print
from exceptions import RAGQueryError, ToolExecutionError


class Agent:
    """Allora RAG Agent using LlamaIndex FunctionAgent architecture"""

    def __init__(
        self,
        index_names: List[str] = [],
        sysprompt: str = default_system_prompt,
        max_tokens: int = 1000,
        logger: logging.Logger = logging.getLogger("uvicorn.error"),
        enable_chart_generation: bool = False,
    ):
        self.logger = logger
        self.chart_generation_enabled = enable_chart_generation
        
        tools: list[BaseTool] = []

        # Create tools from query engines
        tools.extend(create_rag_tools(index_names=index_names, max_tokens=max_tokens))

        tools.append(image_tool)
        
        # Add chart generation tool if enabled
        if enable_chart_generation:
            try:
                chart_tool = create_chart_tool()
                tools.append(chart_tool)
                self.logger.info("Chart generation tool added to agent")
            except ImportError as e:
                self.logger.warning(f"Chart generation requested but dependencies not available: {e}")
        
        # Create the function calling agent
        self.agent = FunctionCallingAgent.from_tools(
            tools=tools,
            # llm=OpenAI(
            llm=OpenAIResponses(
                built_in_tools=[{"type": "web_search_preview"}], #, {"type": "image_generation"}],
                model="gpt-4o",
                temperature=0.3,  # Lower temperature for more focused, consistent responses
                max_tokens=max_tokens * 2,  # Increase max tokens for more verbose responses
                reuse_client=True,
            ),
            system_prompt=sysprompt,
            memory=ChatMemoryBuffer.from_defaults(token_limit=8000),
            verbose=True,
        )
        
        self.logger.info(f"Agent initialized with {len(tools)} tools: {[tool.metadata.name for tool in tools]}")

    async def answer_allora_query(self, request_id: int, message: str) -> Tuple[str, List[str], List[str]]:
        """Answer query using FunctionAgent with source extraction and chart generation
        
        Returns:
            Tuple of (answer, sources, image_paths)
        """
        try:
            self.logger.info(f"[{request_id}] Processing query: '{message[:50]}{'...' if len(message) > 50 else ''}'")
            
            # Use the agent to process the query
            self.logger.info(f"[{request_id}] Calling agent.achat() with message: '{message}'")
            response = await self.agent.achat(message)
            self.logger.info(f"[{request_id}] Agent.achat() completed successfully")

            pretty_print(response)
            
            # Debug logging for response structure
            self.logger.info(f"[{request_id}] Response type: {type(response)}")
            self.logger.info(f"[{request_id}] Response attributes: {[attr for attr in dir(response) if not attr.startswith('_')]}")
            if hasattr(response, 'response'):
                self.logger.info(f"[{request_id}] response.response: {response.response}")
            if hasattr(response, 'chat_history'):
                self.logger.info(f"[{request_id}] chat_history length: {len(response.chat_history) if response.chat_history else 0}")
            if hasattr(response, 'sources'):
                self.logger.info(f"[{request_id}] sources: {response.sources}")
            if hasattr(response, 'source_nodes'):
                self.logger.info(f"[{request_id}] source_nodes length: {len(response.source_nodes) if response.source_nodes else 0}")
            
            # Extract the response text with fallback logic
            answer = self._extract_response_text(response, request_id)
            self.logger.info(f"[{request_id}] Final extracted answer: '{answer[:100]}{'...' if len(answer) > 100 else ''}'")
            
            # Extract sources from the response
            sources = self._extract_sources_from_response(response)

            # Extract artifacts from response
            image_paths = self._extract_images_from_response(response)
            self.logger.info(f"[{request_id}] Extracted image paths: {image_paths}")
            
            # # Check if chart was generated (look for file path in response)
            # image_path = self._extract_image_path_from_response(response)
            
            self.logger.info(f"[{request_id}] Query completed - Response length: {len(answer)}, Sources: {len(sources)}, Image: {bool(image_paths and len(image_paths) > 0)}")
            
            return answer, sources, image_paths
            
        except ToolExecutionError as e:
            self.logger.error(f"[{request_id}] Tool execution error: {str(e)}")
            return "I encountered an issue with one of my tools while processing your request. Please try again.", [], []
        except RAGQueryError as e:
            self.logger.error(f"[{request_id}] RAG query error: {str(e)}")
            return "I had trouble accessing my knowledge base. Please try rephrasing your question.", [], []
        except Exception as e:
            self.logger.error(f"[{request_id}] Unexpected error processing query: {str(e)}")
            return "I encountered an unexpected error. Please try again later.", [], []

    def _extract_response_text(self, response, request_id: int) -> str:
        """Extract response text with robust fallback logic"""
        
        # Try multiple potential response attributes in order of preference
        potential_attributes = [
            'response',      # Standard LlamaIndex response
            'content',       # Alternative response format
            'text',          # Text-based response
            'message',       # Message-based response
            'output',        # Output-based response
        ]
        
        for attr in potential_attributes:
            try:
                if hasattr(response, attr):
                    value = getattr(response, attr)
                    if value is not None:
                        text = str(value).strip()
                        if text and text != "None":
                            self.logger.info(f"[{request_id}] Extracted response from '{attr}' attribute")
                            return text
            except Exception as e:
                self.logger.debug(f"[{request_id}] Error accessing attribute '{attr}': {str(e)}")
                continue
        
        # If no standard attributes work, try to convert the whole response
        try:
            response_str = str(response).strip()
            if response_str and response_str != "None" and not response_str.startswith("<"):
                self.logger.warning(f"[{request_id}] Using fallback string conversion of response")
                return response_str
        except Exception as e:
            self.logger.debug(f"[{request_id}] Error with fallback string conversion: {str(e)}")
        
        # Ultimate fallback
        self.logger.error(f"[{request_id}] Could not extract response text from any attribute")
        return "I apologize, but I couldn't generate a proper response. Please try again."

    def _extract_images_from_response(self, response) -> list[str]:
        """Extract image URLs and file paths from agent tool call results"""
        image_paths: list[str] = []
        
        try:
            # Method 1: Check for ImageBlock objects in sources (legacy)
            if hasattr(response, 'sources') and response.sources:
                for source in response.sources:
                    if isinstance(source, ImageBlock) and source.url:
                        image_paths.append(source.url)
                        self.logger.info(f"Found ImageBlock URL: {source.url}")
            
            # Method 2: Parse tool call results from chat history
            if hasattr(response, 'chat_history'):
                for message in response.chat_history:
                    if hasattr(message, 'additional_kwargs') and message.additional_kwargs:
                        tool_calls = message.additional_kwargs.get('tool_calls', [])
                        for tool_call in tool_calls:
                            if tool_call.get('function', {}).get('name') in ['image_generation', 'generate_chart']:
                                # Extract from tool call result
                                result = self._extract_from_tool_call_result(tool_call)
                                if result:
                                    image_paths.extend(result)
            
            # Method 3: Check response sources for tool outputs (ToolOutput objects)
            if hasattr(response, 'sources') and response.sources:
                for source in response.sources:
                    if hasattr(source, 'tool_name') and source.tool_name == 'generate_chart':
                        # Extract from chart tool output
                        if hasattr(source, 'raw_output') and source.raw_output:
                            raw_output = str(source.raw_output)
                            if raw_output.endswith('.png') or raw_output.endswith('.jpg'):
                                image_paths.append(raw_output)
                                self.logger.info(f"Found chart file from ToolOutput: {raw_output}")
                    elif hasattr(source, 'tool_name') and source.tool_name == 'image_generation':
                        # Extract from DALL-E tool output  
                        if hasattr(source, 'raw_output') and source.raw_output:
                            raw_output = source.raw_output
                            if isinstance(raw_output, list):
                                for url in raw_output:
                                    if isinstance(url, str) and url.startswith('http'):
                                        image_paths.append(url)
                                        self.logger.info(f"Found DALL-E URL from ToolOutput: {url}")
                            elif isinstance(raw_output, str) and raw_output.startswith('http'):
                                image_paths.append(raw_output)
                                self.logger.info(f"Found DALL-E URL from ToolOutput: {raw_output}")
            
            # Method 4: Check response source_nodes for tool outputs (legacy support)
            if hasattr(response, 'source_nodes'):
                for node in response.source_nodes:
                    if hasattr(node, 'metadata'):
                        metadata = node.metadata
                        if 'tool_output' in metadata:
                            result = self._parse_tool_output_for_images(metadata['tool_output'])
                            if result:
                                image_paths.extend(result)
            
            # Method 5: Parse the response text for image URLs and file paths
            if hasattr(response, 'response'):
                text_images = self._extract_images_from_text(str(response.response))
                image_paths.extend(text_images)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_images = []
            for img in image_paths:
                if img not in seen and img:
                    seen.add(img)
                    unique_images.append(img)
                    
            self.logger.info(f"Total unique images extracted: {len(unique_images)}")
            return unique_images
            
        except Exception as e:
            self.logger.error(f"Error extracting images from response: {str(e)}")
            return []
    
    def _extract_from_tool_call_result(self, tool_call) -> list[str]:
        """Extract image URLs/paths from a specific tool call result"""
        images = []
        try:
            function_name = tool_call.get('function', {}).get('name')
            if function_name == 'image_generation':
                # DALL-E returns list of URLs
                result = tool_call.get('result', [])
                if isinstance(result, list):
                    for url in result:
                        if isinstance(url, str) and (url.startswith('http') or url.startswith('https')):
                            images.append(url)
                            self.logger.info(f"Found DALL-E image URL: {url}")
            elif function_name == 'generate_chart':
                # Chart tool returns file path
                result = tool_call.get('result', '')
                if isinstance(result, str) and (result.endswith('.png') or result.endswith('.jpg')):
                    images.append(result)
                    self.logger.info(f"Found chart file path: {result}")
        except Exception as e:
            self.logger.debug(f"Error parsing tool call result: {str(e)}")
        return images
    
    def _parse_tool_output_for_images(self, tool_output) -> list[str]:
        """Parse tool output text for image URLs and file paths"""
        images = []
        try:
            if isinstance(tool_output, str):
                # Look for URLs
                import re
                url_pattern = r'https?://[^\s<>"\']+\.(?:png|jpg|jpeg|gif|webp)'
                urls = re.findall(url_pattern, tool_output)
                images.extend(urls)
                
                # Look for file paths
                path_pattern = r'/[^\s<>"\']*\.(?:png|jpg|jpeg|gif)'
                paths = re.findall(path_pattern, tool_output)
                images.extend(paths)
                
            elif isinstance(tool_output, list):
                for item in tool_output:
                    if isinstance(item, str) and ('http' in item or '.png' in item or '.jpg' in item):
                        images.append(item)
        except Exception as e:
            self.logger.debug(f"Error parsing tool output: {str(e)}")
        return images
    
    def _extract_images_from_text(self, text: str) -> list[str]:
        """Extract image URLs and file paths from response text"""
        images = []
        try:
            import re
            # Pattern for URLs
            url_pattern = r'https?://[^\s<>"\']+\.(?:png|jpg|jpeg|gif|webp)'
            urls = re.findall(url_pattern, text)
            images.extend(urls)
            
            # Pattern for file paths
            path_pattern = r'/tmp/[^\s<>"\']*\.(?:png|jpg|jpeg|gif)'
            paths = re.findall(path_pattern, text)
            images.extend(paths)
            
            if images:
                self.logger.info(f"Extracted images from text: {images}")
                
        except Exception as e:
            self.logger.debug(f"Error extracting images from text: {str(e)}")
        return images

    def _extract_sources_from_response(self, response) -> List[str]:
        """Extract sources from the agent response"""
        sources = []
        
        try:
            # Check if response has source_nodes attribute
            if hasattr(response, 'source_nodes') and response.source_nodes:
                sources = self._extract_sources_from_nodes(response.source_nodes)
            
            # Also check for sources in tool outputs
            if hasattr(response, 'sources') and response.sources:
                for source in response.sources:
                    if hasattr(source, 'source_nodes') and source.source_nodes:
                        sources.extend(self._extract_sources_from_nodes(source.source_nodes))
            
            # Remove duplicates while preserving order
            seen = set()
            unique_sources = []
            for source in sources:
                if source not in seen:
                    seen.add(source)
                    unique_sources.append(source)
            
            self.logger.debug(f"Extracted {len(unique_sources)} unique sources")
            return unique_sources
            
        except Exception as e:
            self.logger.warning(f"Error extracting sources: {str(e)}")
            return []

    def _extract_sources_from_nodes(self, source_nodes) -> List[str]:
        """Extract source information from source nodes"""
        sources = []
        
        for node_with_score in source_nodes:
            try:
                # Handle both NodeWithScore and direct Node objects
                node = getattr(node_with_score, "node", node_with_score)
                metadata = getattr(node, "metadata", {})
                
                # Try multiple metadata fields in order of preference
                source_candidates = [
                    metadata.get("source"),
                    metadata.get("file_name"), 
                    metadata.get("filename"),
                    metadata.get("file_path"),
                    metadata.get("document_title"),
                    metadata.get("title"),
                    metadata.get("document_id"),
                    metadata.get("page_label"),
                    getattr(node, "id_", None)
                ]
                
                # Use the first non-empty source
                for candidate in source_candidates:
                    if candidate and str(candidate).strip():
                        sources.append(str(candidate).strip())
                        break
                        
            except Exception as e:
                self.logger.debug(f"Error processing source node: {str(e)}")
                continue
                
        return sources

    def _extract_image_path_from_response(self, response) -> str:
        """Extract image path from agent response if chart was generated"""
        try:
            response_text = str(response.response)
            
            # Look for file paths in the response
            import re
            
            # Pattern to match file paths (especially .png files)
            path_patterns = [
                r'/tmp/[^\\s]+\\.png',
                r'/[^\\s]+\\.png',
                r'[^\\s]+\\.png'
            ]
            
            for pattern in path_patterns:
                matches = re.findall(pattern, response_text)
                if matches:
                    # Return the first valid file path found
                    for match in matches:
                        if os.path.exists(match):
                            return match
            
            self.logger.debug(f"Found no image path")
            return None
            
        except Exception as e:
            self.logger.debug(f"Error extracting image path: {str(e)}")
            return None

    @property 
    def image_generation_enabled(self) -> bool:
        """Backward compatibility property"""
        return self.chart_generation_enabled

    # # Keep these properties for backward compatibility with main.py and slack.py
    # def __getattr__(self, name):
    #     """Provide backward compatibility for accessing query_engines"""
    #     if name == 'query_engines':
    #         return self.query_engines
    #     raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")