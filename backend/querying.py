import os
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_chroma import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()


# Request/Response Models
class Query(BaseModel):
    """Query model for user questions."""
    text: str = Field(..., description="The question to answer", min_length=1)


class Response(BaseModel):
    """Response model for RAG answers."""
    query: str = Field(..., description="The original question")
    answer: str = Field(..., description="The generated answer")


# Configuration
COLLECTION_NAME = "my_collection"
PERSIST_DIRECTORY = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gemini-2.5-flash"
LLM_TEMPERATURE = 0.0
RETRIEVER_K = 2


# Initialize components
def get_embedding_function() -> SentenceTransformerEmbeddings:
    """Initialize and return the embedding function."""
    return SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)


def get_vectorstore() -> Chroma:
    """Initialize and return the Chroma vector store."""
    embedding_function = get_embedding_function()
    
    if not os.path.exists(PERSIST_DIRECTORY):
        raise FileNotFoundError(
            f"Vector store directory not found: {PERSIST_DIRECTORY}. "
            "Please run indexing.py first to create the vector store."
        )
    
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_function,
        persist_directory=PERSIST_DIRECTORY
    )


def get_llm() -> ChatGoogleGenerativeAI:
    """Initialize and return the LLM."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found in environment variables. "
            "Please set it in your .env file."
        )
    
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE
    )


# Initialize global components
try:
    embedding_function = get_embedding_function()
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K}
    )
    llm = get_llm()
except Exception as e:
    print(f"Warning: Failed to initialize components: {e}")
    embedding_function = None
    vectorstore = None
    retriever = None
    llm = None


# Prompt template
PROMPT_TEMPLATE = """Answer the question based only on the following context. 
If the context doesn't contain enough information to answer the question, 
say that you don't have enough information.

Context:
{context}

Question: {question}

Answer: """

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)


# Helper functions
def format_documents(docs: List[Document]) -> str:
    """Format a list of documents into a single string."""
    return "\n\n".join(
        f"Document {i+1}:\n{doc.page_content}" 
        for i, doc in enumerate(docs)
    )


def create_rag_chain():
    """Create and return the RAG chain."""
    if not retriever or not llm:
        raise RuntimeError("RAG components not initialized properly")
    
    return (
        {
            "context": retriever | format_documents,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )


# API Endpoints
@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "message": "RAG API with LangChain",
        "status": "running",
        "endpoints": {
            "chat": "/chatting",
            "health": "/health"
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    status = {
        "status": "healthy",
        "vectorstore": vectorstore is not None,
        "llm": llm is not None,
        "retriever": retriever is not None
    }
    
    if not all([vectorstore, llm, retriever]):
        status["status"] = "degraded"
        return status
    
    return status


@app.post("/chatting", response_model=Response)
def get_response(query: Query):
    """
    Process a query using RAG and return an answer.
    
    Args:
        query: The query object containing the question text
        
    Returns:
        Response object with the answer and metadata
        
    Raises:
        HTTPException: If components are not initialized or processing fails
    """
    if not retriever or not llm:
        raise HTTPException(
            status_code=503,
            detail="RAG components not initialized. Please check the health endpoint."
        )
    
    try:
        # Create RAG chain
        rag_chain = create_rag_chain()
        
        # Invoke the chain
        answer = rag_chain.invoke(query.text)
        
        return Response(
            query=query.text,
            answer=answer,
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )
