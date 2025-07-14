import os
from llama_index.core.tools import BaseTool, QueryEngineTool
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex
from llama_index.llms.openai import OpenAI
from sysprompt import default_system_prompt


indices = {
    "alloradocs": {
        "description": "Search and retrieve information from the public-facing Allora documentation",
    },
    "allora_chain": {
        "description": "Search and retrieve information from open-source Allora code",
    },
    "allora_production": {
        "description": "Search and retrieve information from open-source Allora code",
    }
}

def create_rag_tools(index_names: list[str], max_tokens: int) -> list[BaseTool]:
    tools: list[BaseTool] = []
    for name in index_names:
        qe = LlamaCloudIndex(
            name=name,
            project_name="Default",
            api_key=os.environ["LLAMA_CLOUD_API_KEY"],
            organization_id=os.environ["LLAMA_CLOUD_ORG_ID"],
        ).as_query_engine(
            use_async=True,
            similarity_top_k=5,
            llm=OpenAI(
                model="gpt-4o",
                temperature=0.5,
                max_tokens=max_tokens,
                system_prompt=default_system_prompt,
                reuse_client=False,
            ),
        )

        tool = QueryEngineTool.from_defaults(
            query_engine=qe,
            name=name,
            description=indices[name]["description"],
        )

        tools.append(tool)

    return tools

