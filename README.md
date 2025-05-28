# Allora RAG Information Bot

We're building a RAG Chatbot powered by **GPT-4** to answer questions about Allora Labs using content from our documentation, research papers, and codebase. The chatbot is accessible via Slack, Discord, and our documentation website.

---

Documention needs to be updated (4/23/25)

## Description

Our Q&A chat project uses a LlamaCloud vector database called **allora_production**. Data is converted into **3072-dimensional vectors** using the OpenAI `text-embedding-3-large` model. Our LlamaIndex connects this database with GPT-4o. When a user submits a question via a FastAPI POST endpoint, the following happens:

1. **Receive Question:** The user's question is received as plain text.
2. **Convert to Vector:** The question is embedded into a vector.
3. **Generate Response:** GPT-4o, informed by context retrieved from LlamaCloud, generates a helpful answer.

This workflow effectively bridges the user's query with GPT-4o by gathering relevant background information from our stored context.

---

## Training Data (Updated 2/20/25)

### Major Research Papers
- [Allora White Paper](https://www.allora.network/research/optimizing-decentralized-online-learning-for-supervised-regression-and-classification-problems)
- [Merit-based Sortition in Decentralized Systems](https://www.allora.network/research/merit-based-sortition-in-decentralized-systems)
- [Optimizing Decentralized Online Learning for Supervised Regression and Classification Problems](https://www.allora.network/research/optimizing-decentralized-online-learning-for-supervised-regression-and-classification-problems)

### Major Repos/Readmes
- [allora-chain](https://github.com/allora-network/allora-chain)
- [allora-offchain-node](https://github.com/allora-network/allora-offchain-node)
- [coin prediction worker](https://github.com/allora-network/basic-coin-prediction-node)
- [coin prediction reputer](https://github.com/allora-network/coin-prediction-reputer)
- [autogluon-prediction](https://github.com/allora-network/autogluon-prediction)
- [allora-docs](https://github.com/allora-network/docs)

### Other Documentation
- [CometBFT Docs](https://docs.cometbft.com/v0.38/)
- [Cosmos Docs](https://github.com/cosmos/cosmos-sdk-docs)

---

## Updating the Knowledge Context

To add new documentation, you must add additional data embeddings to the **allora_production** Llamacloud index. Follow these steps:

1. **Split & Vectorize:**  
   Use the LangChain library (or another method that adheres to 3072 dimensions and uses `text-embedding-3-large`) to split and vectorize your data.

2. **Store in Pinecone:**  
   Ensure the vectorized data is stored in Pinecone.

3. **Automatic Update:**  
   Any new embeddings are automatically included in responses. Verify insertion by searching for specific keys (e.g., an ID or source) in Pinecone.

> **Note:** Adjust `chunk_size` and `chunk_overlap` to control the granularity of your data representation.

---
## Example: Splitting and Vectorizing PDF

**Process Overview:**

- **Load PDF:**  
  Uses `PyMuPDFLoader` to load a PDF from a local path.

- **Split Document:**  
  Uses `RecursiveCharacterTextSplitter` to break the document into chunks.

- **Generate Embeddings & Vector Store:**  
  Creates text embeddings with OpenAI and stores them in a Pinecone vector store.

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

vector_store = PineconeVectorStore.from_documents(
    documents=split_docs,
    embedding=embeddings,
    index_name="alloraproduction"
)
```

## Example: Splitting and Vectorizing GitHub Files

### Process Overview

- **Load Documents:**  
  Clones a GitHub repository and filters for Markdown files using `GitLoader`.

- **Split Documents:**  
  Uses a language-aware splitter (`RecursiveCharacterTextSplitter.from_language`) configured for Python to break the documents into manageable chunks.

- **Generate Embeddings & Store Vectors:**  
  Converts the document chunks into vectors using the OpenAI `text-embedding-3-large` model and stores them in a Pinecone index.

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

# change based on repo you want to copy from 
repo_url = "https://github.com/allora-network/allora-offchain-node/"

# allora-offchain-node used as an example, change based on what repo you decide to vectorize
repo_path = "allora-offchain-node"

# change based on branch you want to copy from 
branch = "dev"

loader = GitLoader(
    clone_url=repo_url,
    repo_path=repo_path,
    branch=branch,
        file_filter=lambda file_path: file_path.endswith((".go", ".py"))

)# Load documents from the repository
documents = loader.load()


# Update metadata for each document with repo info and other details
for doc in documents:
    # Assuming GitLoader stores the file path in the "source" metadata field
    file_path = doc.metadata.get("source", "")
    if file_path:
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_path)[1]
        # Infer language from file extension (could be extended with more sophisticated logic)
        language = "Python" if file_ext == ".py" else "Go" if file_ext == ".go" else "Unknown"
        
        doc.metadata.update({
            "file_name": file_name,
            "file_path": file_path,
            "repository_name": os.path.splitext(os.path.basename(repo_url))[0],
            "branch": branch,
            "file_extension": file_ext,
            "language": language,
            # You could add more fields here, such as commit hash, author, etc.
        })

# 2. Split documents using a language-aware splitter
# Note: Choose the appropriate language or a generic splitter if processing multiple languages.
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,  # Or use a generic splitter if needed
    chunk_size=500,
    chunk_overlap=200,
)
split_docs = splitter.split_documents(documents)

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

## Example: Testing the Allora Chatbot Server

### Process Overview

- **Define the Server URL:**  
  Set the endpoint URL for your chatbot.

- **Prepare the Request Payload:**  
  Create a JSON object with your question.

- **Send the Request:**  
  POST the payload using Python's `requests` library.

- **Handle the Response:**  
  Print the chatbot's response and its sources.

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
Sources: ['/markdown_files4/pages/devs/reference/module-accounts.mdx', '/markdown_files4/pages/home/overview.mdx']*

## Future Updates

When new data is added, update this document to keep track of changes and ensure the knowledge context remains current.



