# Allora RAG Information Bot

We're developing a RAG Chatbot powered by GPT-4 that will answer questions about Allora Labs using content from our documentation, research papers, and portions of our codebase. The chatbot will be accessible via Slack, Discord, and our documentation website.


## Description 

We've set up a Pinecone vector database called alloraproduction for our Q&A chat project. Our data is transformed into 3072-dimensional vectors using the OpenAI text-embedding-3-large model and stored in this database. We then established a LangChain workflow that links our Pinecone database with a GPT-4 instance. When a user sends in a question via a FastAPI POST endpoint, the following steps occur:

1. The user's question is received as a plain text string.
2. The question is converted into a vector using the text-embedding-3-large model.
3. This vector is then sent to our LangChain workflow, which uses the GPT-4 instance—trained on context retrieved from our Pinecone index—to generate a response.
   
Our LangChain workflow acts as a bridge between the user's query and the GPT-4 model. Our model first gathers the right background information and then uses GPT-4 to turn that information into a helpful response based our Pinecone context


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


## How to update the knowledge context/add more documentation?

### Overview

To add more documentation to our Allora agent, we need to add more data (in the form of embeddings) to our Pinecone database 'alloraproduction'. In the examples below, we split and vectorized our data/files into Pinecone using the LangChain library (though any method that adheres to the 3072 dimensions and utilizes the OpenAI text-embedding-3-large model is acceptable). Regardless of the library you use to split and vectorize your data, you must store it within the Pinecone database. 

Once you've added the new data embeddings to your Pinecone database, any prompt sent to your endpoint will automatically include this updated context to answer it. You can confirm that the embeddings have been successfully added by searching for specific key or field values—such as an ID or source—in Pinecone. 

****Examples to add github files and pdf data are shown below****

#### Notes

chunk_size and chunk_overlap are hyperparameters that you can adjust to control how detailed the data representation is when it is searched.

### Example: Splitting and Vectorizing PDF 

_Load a PDF Document:_
- It uses PyMuPDFLoader to load a PDF from a specified local path, resulting in a list of document objects.

_Split the Document into Chunks:_
- The loaded document is split into smaller chunks using RecursiveCharacterTextSplitter. This is useful for processing large texts in manageable segments.

_Create Text Embeddings and Set Up a Vector Store:_
- Text embeddings are generated using the OpenAIEmbeddings model, and these embeddings are then stored in a Pinecone vector store for efficient retrieval.


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

# set/get our server account APIs here
os.environ["OPENAI_API_KEY"] = ""
os.environ["PINECONE_API_KEY"] = ""

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# load our server account API here
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("alloraproduction")

vector_store = PineconeVectorStore(embedding=embeddings, index=index)
```


### Example: Splitting and Vectorizing Github Files

_Load Documents from GitHub:_
- Uses GitLoader to clone the repository and filter for Markdown files from a specified branch.

_Split Documents:_
- Uses a language-aware splitter (RecursiveCharacterTextSplitter.from_language) configured for Python to break the documents into chunks.

_Generate Embeddings & Store Vectors:_
- Initializes the OpenAI embeddings model (text-embedding-3-large).
- Creates a Pinecone vector store by converting document chunks into vectors and storing them under a specified index.

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
    chunk_size=750,
    chunk_overlap=200,
)
split_docs = python_splitter.split_documents(documents)

# 3. Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# 4. Pinecone setup
index_name = "alloraproduction"
vector_dimension = 3072  # Dimension for text-embedding-3-large


# Initialize vector store with the new client
vector_db = PineconeVectorStore.from_documents(
    documents=split_docs,
    embedding=embeddings,
    index_name=index_name,
    pinecone_api_key=os.getenv("PINECONE_API_KEY")  # Add this line
)

print("Vectorization complete. Documents stored in Pinecone.")

```

### Example: Testing the Allora Chatbot Server

This example demonstrates how to test if the Allora chatbot model is running on your server. The code sends a question as a JSON payload to the chatbot endpoint, then prints the chatbot's response along with the sources from which the answer was derived.

#### How It Works
1.) Define the Server URL:
Set the URL of your chatbot endpoint.

2.) Prepare the Request Payload:
Create a JSON object with the key "message" containing your question. In this example, the question is:
"What makes Allora's reward distribution different than others?"

3.) Send the Request:
Use Python's requests library to POST the payload to the server.

4.) Handle the Response:
If successful, the code prints out the chatbot's response message and the sources used.
In case of errors (HTTP or other exceptions), it prints an error message.

```python

import requests

# URL of the chatbot endpoint. Replace with your actual server URL.
url = "https://your-chatbot-endpoint.com/api/chat"

# The payload containing the message/question for the chatbot.
payload = {
    "message": "What makes Allora's reward distribution different than others?"
}

try:
    # Send a POST request to the server with the JSON payload.
    response = requests.post(url, json=payload)
    
    # Raise an error if the request was unsuccessful.
    response.raise_for_status()
    
    # Parse the JSON response.
    data = response.json()
    
    # Output the chatbot response and its sources.
    print("Response:")
    print("Message:", data.get("response"))
    print("Sources:", data.get("sources"))

except requests.exceptions.HTTPError as http_err:
    print(f"HTTP error occurred: {http_err}")
except Exception as err:
    print(f"Other error occurred: {err}")

```

For this particular example, you should expect an output similar to:

*Response:
Message:  Allora's reward distribution is differentiated and based on a carefully designed incentive mechanism that aligns with the interests of the network and allows for continual learning and improvement. 
Sources: ['/markdown_files4/pages/devs/reference/module-accounts.mdx', '/markdown_files4/pages/home/overview.mdx'*

### Future Updates

When adding new data to the model, it should be added to this readme to keep for future reference.



