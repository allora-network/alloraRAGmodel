# Allora RAG Information Bot

We're developing a RAG Chatbot powered by GPT-4 that will answer questions about Allora Labs using content from our documentation, research papers, and portions of our codebase. The chatbot will be accessible via Slack, Discord, and our documentation website.


## What data did we train it on (updated 2/20/25)

- Major research papers:
  - [Allora White Paper](https://www.allora.network/research/optimizing-decentralized-online-learning-for-supervised-regression-and-classification-problems)
  - [Merit-based sortition in Decentralized Systems](https://www.allora.network/research/merit-based-sortition-in-decentralized-systems)
  - [Optimizing Decentralized Online Learning for Supervised Regression and Classification Problems](https://www.allora.network/research/optimizing-decentralized-online-learning-for-supervised-regression-and-classification-problems)

- Major repos/readmes:
  - [chain](https://github.com/allora-network/allora-chain)
  - [off-chain](https://github.com/allora-network/allora-offchain-node)
  - [coin prediction worker](https://github.com/allora-network/basic-coin-prediction-node)
  - [coin prediction reputer](https://github.com/allora-network/coin-prediction-reputer)
  - autogluon-prediction

- Other docs:
  - [CometBFT Docs](https://docs.cometbft.com/v0.38/)
  - [Cosmos Docs](https://github.com/cosmos/cosmos-sdk-docs)


## Spec (updated 2/20/25)

We instantiated a pinecone vector database under the name 'alloraproduction' under the Q&A chat project. Within the database, we have vectorized our data using the openai text-embedding-3-large model resulting in a database of 3072 dimension vectors. We created a langchain model chain where 


We used fastapi to connect to a POST endpoint that:

1.) receives the user's query (question) in string format

2.) Vectorize the user's question using text-embedding-3-large 

3.) sends vectorized question to langchain workflow to get the chatbot to return a response (answer). 

We use the langchain library to instantiate a workflow between the user's question and gpt4 response. The GPT4 instance we use is trained on our pinecone index that is from:



## How to update the knowledge context/add more documentation?

### Overview

To add more documentation to our Allora agent, we need to add more information to our pinecone database 'alloraproduction' (example for github and local pdf shown below)

In the examples below, we split and vectorized our data/files into Pinecone using the LangChain library (though any method that adheres to the 3072 dimensions and utilizes the OpenAI text-embedding-3-large model is acceptable). Regardless of what library you use to split and vectorize your data, you will need to store it within the Pinecone database. 

Once you've added the new data embeddings to your Pinecone database, any prompt sent to your endpoint will automatically include this updated context. You can confirm that the embeddings have been successfully added by searching for specific key or field values—such as an ID or source—in Pinecone.

#### Notes

chunk_size and chunk_overlap are hyperparameters that you can adjust to control how detailed the data representation is when it is searched.

### Example: Splitting and Vectorizing Text Data


```python
# insert local path of your pdf here 

pdf_path = ""
curindex_name="alloraproduction"

loader = PyMuPDFLoader(pdf_path)
docs = loader.load()  # This returns a list of Document objects

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=50
)
split_docs = text_splitter.split_documents(docs)

# load our server account API here
os.environ["OPENAI_API_KEY"] = ""

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Initialize Pinecone vector store
vector_store = PineconeVectorStore.from_documents(
    split_docs,
    embedding=embeddings,
    index_name=curindex_name
)

# load our server account API here
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("alloraproduction")

vector_store = PineconeVectorStore(embedding=embeddings, index=index)
```

### Example: Splitting and Vectorizing Github Files

```python

import os
from langchain_community.document_loaders import GitLoader
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

os.environ["PINECONE_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


# 1. Clone and load Python files from repo

# allora-offchain-node used as an example, change based on what repo you decide to vectorize
repo_path = "allora-offchain-node"

# change based on branch you want to copy from 
branch = "dev"  

loader = GitLoader(
    clone_url="https://github.com/allora-network/allora-offchain-node/",
    repo_path=repo_path,
    branch=branch,
    file_filter=lambda file_path: file_path.endswith(".md")
)

documents = loader.load()

# 2. Split documents using Python-aware splitter
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=800, # hyperparameter
    chunk_overlap=200, # hyperparameter
)

split_docs = python_splitter.split_documents(documents)

# 3. Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# 4. Pinecone setup

index_name = "alloraproduction"
vector_dimension = 3072  # Dimension for text-embedding-3-large

# Initialize Pinecone ####

# 4. Pinecone setup 
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)

# Initialize vector store with the new client
vector_db = PineconeVectorStore.from_documents(
    documents=split_docs,
    embedding=embeddings,
    index_name=index_name,
    pinecone_api_key=os.getenv("PINECONE_API_KEY")  
)

print("Vectorization complete. Documents stored in Pinecone.")


```


