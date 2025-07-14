import os
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
import openai
import requests
import asyncio

# Set up OpenAI API key
# os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
openai.api_key = os.environ["OPENAI_API_KEY"]

# Define the custom image generation tool
def generate_image(prompt: str, model: str = "dall-e-3", n: int = 1, size: str = "1024x1024", quality: str = "standard") -> list[str]:
    """
    Generate images using OpenAI's DALL·E API based on a text prompt.

    Args:
        prompt (str): The text prompt to generate the image.
        model (str): The DALL·E model to use (default: dall-e-3).
        n (int): Number of images to generate (default: 1).
        size (str): Image size (e.g., "1024x1024").
        quality (str): Image quality (e.g., "standard", "hd").

    Returns:
        list[str]: List of generated image URLs.
    """
    try:
        response = openai.images.generate(
            model=model,
            prompt=prompt,
            n=n,
            # size=size,
            # quality=quality
        )
        return [image.url for image in response.data]
    except Exception as e:
        return [f"Error generating image: {str(e)}"]

# Wrap the image generation function as a LlamaIndex tool
image_tool = FunctionTool.from_defaults(
    fn=generate_image,
    name="image_generation",
    description="Generate images from text prompts using OpenAI's DALL·E API. Use this when the user explicitly requests an image or visual output. DO NOT use it for any kinds of charts, graphs, or mathematical output."
)