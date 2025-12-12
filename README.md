# Allora RAG Information Bot

We're building an Agentic RAG Chatbot powered by **GPT-4** to answer questions about Allora Labs using content from our documentation, research papers, and codebase. The chatbot is accessible via Slack, Discord, and our documentation website.

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

## Wizard Tools (MCP Integration)

The RAG system integrates with the Allora Topic Wizard via MCP (Model Context Protocol). This provides 43+ tools for blockchain queries, topic management, and infrastructure operations.

### Architecture

```
RAG System (FunctionAgent)
    │
    └── spawns ──► MCP Server (allora-wizard-mcp)
                       │
                       └── connects to ──► Wizard Backend API
```

### Environment Variables

```bash
# Required for wizard tools
WIZARD_API_URL=http://localhost:3000              # Wizard backend URL
MCP_WIZARD_PACKAGE=@allora-network/wizard-mcp     # npm package name (GitHub Packages)
# For local dev: MCP_WIZARD_PACKAGE=allora-wizard-mcp

# Optional
# WIZARD_API_KEY=your_api_key             # Only if wizard auth enabled
```

### Available Tools

The MCP server provides tools for:
- **Topic queries**: `get_topic`, `topic_exists`, `is_topic_active`, `get_topic_stake`
- **Whitelist management**: `is_worker_whitelisted`, `whitelist_workers`, `whitelist_reputers`
- **Infrastructure**: `get_k8s_deployments`, `get_osm_config`, `get_osm_wallets`
- **Scores & metrics**: `get_inferer_score`, `get_forecaster_score`, `get_reputer_score`

Run the server and check logs to see all 43+ available tools.

---

## Future Updates

When new data is added, update this document to keep track of changes and ensure the knowledge context remains current.



