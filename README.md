# RAG with LangChain

A Retrieval-Augmented Generation (RAG) system built with LangChain, Chroma vector database, and FastAPI. This project enables you to build a question-answering system that retrieves relevant information from your documents and generates accurate answers using Google's Gemini model.

## 🚀 Features

- **Document Processing**: Supports multiple file formats (PDF, DOCX, TXT)
- **Vector Storage**: Uses Chroma for efficient document embedding and retrieval
- **RAG Pipeline**: Implements a complete RAG workflow with document chunking, embedding, and retrieval
- **REST API**: FastAPI-based API for easy integration
- **Google Gemini Integration**: Uses Google's Gemini 2.5 Flash model for answer generation
- **Sentence Transformers**: Utilizes state-of-the-art embeddings for semantic search

## 📋 Prerequisites

- Python 3.10 or higher
- Google API Key (for Gemini model)
- Documents to process (PDF, DOCX, or TXT files)

## 🛠️ Installation

1. **Clone the repository** (or navigate to the project directory):
   ```bash
   cd rag-langchain
   ```

2. **Install dependencies**:
   
   Using `uv` (recommended):
   ```bash
   uv pip install -e .
   ```
   
   Or using `pip`:
   ```bash
   pip install -e .
   ```

3. **Set up environment variables**:
   
   Create a `.env` file in the root directory:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

## 📁 Project Structure

```
rag-langchain/
├── backend/
│   ├── indexing.py      # Document indexing and vector store creation
│   └── querying.py      # FastAPI server for querying
├── sample_data/         # Sample documents for testing
├── chroma_db/          # Chroma vector database (created after indexing)
├── pyproject.toml      # Project dependencies
└── README.md           # This file
```

## 🚀 Usage

### Step 1: Index Your Documents

First, place your documents (PDF, DOCX, or TXT files) in the `sample_data` folder (or modify the path in `indexing.py`).

Run the indexing script to process documents and create the vector store:

```bash
python backend/indexing.py
```

This will:
- Load all documents from the `sample_data` folder
- Split them into chunks (500 characters with 50 character overlap)
- Generate embeddings using Sentence Transformers
- Store them in Chroma vector database at `./chroma_db`

### Step 2: Start the API Server

Start the FastAPI server:

```bash
uvicorn backend.querying:app --reload
```

The API will be available at `http://localhost:8000`

### Step 3: Query the API

#### Using cURL:

```bash
curl -X POST "http://localhost:8000/chatting" \
  -H "Content-Type: application/json" \
  -d '{"text": "What is artificial intelligence?"}'
```

#### Using Python:

```python
import requests

response = requests.post(
    "http://localhost:8000/chatting",
    json={"text": "What is artificial intelligence?"}
)

print(response.json())
```

#### Using the Interactive API Docs:

Visit `http://localhost:8000/docs` in your browser to access the interactive Swagger UI documentation.

## 📡 API Endpoints

### `GET /`

Root endpoint with API information.

**Response:**
```json
{
  "message": "RAG API with LangChain",
  "status": "running",
  "endpoints": {
    "chat": "/chatting",
    "health": "/health"
  }
}
```

### `GET /health`

Health check endpoint to verify system status.

**Response:**
```json
{
  "status": "healthy",
  "vectorstore": true,
  "llm": true,
  "retriever": true
}
```

### `POST /chatting`

Query the RAG system with a question.

**Request Body:**
```json
{
  "text": "Your question here"
}
```

**Response:**
```json
{
  "query": "Your question here",
  "answer": "Generated answer based on retrieved context"
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/chatting" \
  -H "Content-Type: application/json" \
  -d '{"text": "How did the invention of agriculture influence the development of early human societies?"}'
```

## ⚙️ Configuration

You can modify the configuration in `backend/querying.py`:

```python
COLLECTION_NAME = "my_collection"        # Chroma collection name
PERSIST_DIRECTORY = "./chroma_db"       # Vector store directory
EMBEDDING_MODEL = "all-MiniLM-L6-v2"    # Embedding model
LLM_MODEL = "gemini-2.5-flash"          # LLM model
LLM_TEMPERATURE = 0.0                    # LLM temperature
RETRIEVER_K = 2                          # Number of documents to retrieve
```

In `backend/indexing.py`, you can adjust:

```python
chunk_size=500      # Document chunk size
chunk_overlap=50    # Overlap between chunks
```

## 🔧 Technologies Used

- **LangChain**: Framework for building LLM applications
- **Chroma**: Vector database for embeddings
- **FastAPI**: Modern web framework for building APIs
- **Sentence Transformers**: State-of-the-art sentence embeddings
- **Google Gemini**: Large language model for answer generation
- **Pydantic**: Data validation using Python type annotations

## 📝 Supported File Formats

- **PDF** (`.pdf`) - Using PyPDFLoader
- **Word Documents** (`.docx`) - Using Docx2txtLoader
- **Text Files** (`.txt`) - Using TextLoader

## 🐛 Troubleshooting

### Vector Store Not Found

If you get an error about the vector store not being found:
1. Make sure you've run `backend/indexing.py` first
2. Check that the `chroma_db` directory exists
3. Verify the `PERSIST_DIRECTORY` path in `querying.py`

### API Key Issues

If you encounter API key errors:
1. Ensure your `.env` file exists in the root directory
2. Verify `GOOGLE_API_KEY` is set correctly
3. Check that the API key is valid and has the necessary permissions

### Import Errors

If you get import errors:
1. Make sure all dependencies are installed: `pip install -e .`
2. Verify you're using Python 3.10 or higher
3. Check that all required packages are in `pyproject.toml`

## 📚 Example Workflow

1. **Prepare Documents**: Add your PDF, DOCX, or TXT files to `sample_data/`

2. **Index Documents**:
   ```bash
   python backend/indexing.py
   ```

3. **Start Server**:
   ```bash
   uvicorn backend.querying:app --reload
   ```

4. **Query the API**:
   ```bash
   curl -X POST "http://localhost:8000/chatting" \
     -H "Content-Type: application/json" \
     -d '{"text": "What is climate change?"}'
   ```
