# How to update the knowledge base of the Allora information agent?

### Overview

We instantiated a Pinecone vector database under the name **`alloraproduction`** within the Q&A chat project. Within the database, we have vectorized our data using the **OpenAI text-embedding-3-large** model, resulting in a database of **3072-dimensional vectors**.

### Example: Splitting and Vectorizing Text Data

We split and vectorized our text data into Pinecone using the LangChain library (though any method that adheres to the 3072 dimensions and utilizes the OpenAI text-embedding-3-large model is acceptable).

```python
pdf_path = ""

loader = PyMuPDFLoader(pdf_path)
docs = loader.load()  # This returns a list of Document objects

os.environ["PINECONE_API_KEY"] = ""

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=50
)
split_docs = text_splitter.split_documents(docs)

os.environ["OPENAI_API_KEY"] = ""

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Initialize Pinecone vector store
vector_store = PineconeVectorStore.from_documents(
    split_docs,
    embedding=embeddings,
    index_name="alloraproduction"  
)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("alloraproduction")

vector_store = PineconeVectorStore(embedding=embeddings, index=index)
