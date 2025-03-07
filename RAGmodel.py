from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
from pinecone import Pinecone
from langchain_community.llms import OpenAI
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_openai import OpenAIEmbeddings  # Updated import
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


# Initialize FastAPI app
app = FastAPI()

# Configure CORS middleware -> this allows api to be called with react frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, allow all origins. In production, restrict this.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


with open("additional_context.txt", "r") as file:
    more_context = file.read()


# set/load environment variables -> edit this for production 
# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Set up components
INDEX_NAME = "alloraproduction"
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"),
                              model="text-embedding-3-large")

# Connect to existing Pinecone index
try:
    # Get Pinecone index object
    index = pc.Index(INDEX_NAME)
    
    # Create vector store directly with index
    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="text"  # Match your index's text field key
    )
except Exception as e:
    raise RuntimeError(f"Error connecting to Pinecone index: {str(e)}")


custom_prompt = PromptTemplate(
    template=(
        "You are a highly knowledgeable assistant who helps users learn about Allora and all of its offerings. Use the context provided below to answer the question. Here is some additional context about the revenue model:"
        f"{more_context}\n\n"
        "If someone asks a question related to cost per inference, this is a non-answerable question. If the answer is not contained in the context, say 'I don't know'.\n\n"
        "Question: {{question}}\n\n"
        "Retrieved Context:\n{{context}}\n\n"
        "Answer:"
    ),
    input_variables=["question", "context"]
)

qa = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY")),
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": custom_prompt}  # use your custom prompt
)


# Request/Response models
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    sources: List[str]

@app.get("/")
async def root():
    return {"Ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        result = qa.invoke({"query": request.message})
        sources = list(set([doc.metadata.get("source", "") for doc in result["source_documents"]]))
        return {
            "response": result["result"],
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT",8000))) # may need to edit this for production
