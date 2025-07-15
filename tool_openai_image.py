import os
from llama_index.core.tools import FunctionTool
import openai
from config import get_config

# Set up OpenAI API key
config = get_config()
openai.api_key = config.openai_api_key

# Define the custom image generation tool
def generate_image(prompt: str, model: str="dall-e-3", n: int=1, size: str="1024x1024", quality: str="standard") -> list[str]:
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