
## Description

We are creating a RAG Chatbot that runs on GPT-4 and answers questions about Allora labs, pulling from information provided by:

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


- Cosmos docs:
  - [CometBFT Docs](https://docs.cometbft.com/v0.38/)
    
this bot should run on Slack, discord, and our doc website page.

# How to update the knowledge base of the Allora information agent?


### Overview

We instantiated a Pinecone vector database under the name **`alloraproduction`** within the Q&A chat project. Within the database, we have vectorized our data using the **OpenAI text-embedding-3-large** model, resulting in a database of **3072-dimensional vectors**.

### Notes

chunk_size and chunk_overlap are hyperparameters that are set depending on how detailed you want the data to be represented when it gets searched



### Example: Splitting and Vectorizing Text Data

We split and vectorized our text data into Pinecone using the LangChain library (though any method that adheres to the 3072 dimensions and utilizes the OpenAI text-embedding-3-large model is acceptable). After completing this step to put our new data embeddings into our pinecone database, it will automatically pull from the newly provided information.



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

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("alloraproduction")

vector_store = PineconeVectorStore(embedding=embeddings, index=index) '''

