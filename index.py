import asyncio
import logging
import os
from llama_index.core import SimpleDirectoryReader
from llama_cloud.client import AsyncLlamaCloud, LlamaCloud
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex
from llama_parse import LlamaParse, ResultType

INDEX_NAME = 'allora_offchain_node'


async def create_index(index_name: str, file_root: str):
    documents = SimpleDirectoryReader(
        input_dir=file_root,
        # input_files=file_paths,
        exclude=['*.git', 'mock_*.go'],
        exclude_hidden=True,
        exclude_empty=True,
        # errors: str = "ignore",
        recursive=True,
        # encoding: str = "utf-8",
        filename_as_id=True,
        required_exts=['go', 'md', 'txt', 'pdf']
        # file_extractor: Optional[dict[str, BaseReader]] = None,
        # num_files_limit: Optional[int] = None,
        # file_metadata: Optional[Callable[[str], dict]] = None,
        # raise_on_error: bool = False,
        # fs: fsspec.AbstractFileSystem | None = None,
    ).load_data()

    index = LlamaCloudIndex.from_documents(
        documents,
        index_name,
        project_name="Default",
        api_key=os.getenv('LLAMA_CLOUD_API_KEY'),
        organization_id=os.getenv('LLAMA_CLOUD_ORGANIZATION_ID'),
        verbose=True,
    )
    print(index.id)
    print(index.index_id)

    # index = await LlamaCloudIndex.acreate_index(
    #     INDEX_NAME,
    #     organization_id=os.getenv('LLAMA_CLOUD_ORGANIZATION_ID'),
    #     api_key=os.getenv('LLAMA_CLOUD_API_KEY'),

    #     documents,
    #     "my_first_index",
    #     project_name="Default",
    #     api_key=os.getenv('LLAMA_CLOUD_API_KEY'),
    #     verbose=True,
    # )

    # client = AsyncLlamaCloud(
    #     token=os.getenv('LLAMA_CLOUD_API_KEY')
    #     # base_url=os.getenv('LLAMA_CLOUD_BASE_URL')
    # )
    # file = client.files.upload_file(upload_file=file, project_id='fc533996-6a15-434a-b652-36661862fdd1', )
    # pipeline_files = client.pipelines.add_files_to_pipeline(pipeline_id, request=[{'file_id': file.id}])


if __name__ == "__main__":
    # env vars
    os.environ["LLAMA_CLOUD_API_KEY"] = os.getenv("LLAMA_CLOUD_API_KEY", '')
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", '')

    # logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    asyncio.gather(
        create_index('allora_offchain_node', '~/projects/allora/allora-offchain-node')
    )
