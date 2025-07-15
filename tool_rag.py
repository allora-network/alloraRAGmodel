import os
from llama_index.core.tools import BaseTool, QueryEngineTool
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex
from llama_index.llms.openai import OpenAI
from sysprompt import default_system_prompt
from config import get_config


indices = {
    "alloradocs": {
        "description": "Search and retrieve information from the public-facing Allora documentation",
    },
    "allora_production": {
        "description": "Search and retrieve information from open-source Allora code",
    },
    "enablement": {
        "description": "Search and retrieve information about internal processes, procedures, initiatives, and goals at Allora Labs"
    },
}

def create_rag_tools(index_names: list[str], max_tokens: int) -> list[BaseTool]:
    config = get_config()
    tools: list[BaseTool] = []
    
    for name in index_names:
        qe = LlamaCloudIndex(
            name=name,
            project_name="Default",
            api_key=config.llama_cloud_api_key,
            organization_id=config.llama_cloud_org_id,
        ).as_query_engine(
            use_async=True,
            similarity_top_k=config.rag.similarity_top_k,
            llm=OpenAI(
                model=config.agent.model,
                temperature=config.rag.temperature,
                max_tokens=max_tokens,
                system_prompt=default_system_prompt,
                reuse_client=config.agent.reuse_client,
            ),
        )

        tool = QueryEngineTool.from_defaults(
            query_engine=qe,
            name=name,
            description=indices[name]["description"],
        )

        tools.append(tool)

    return tools

