import os
from typing import List
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

def load_documents(folder_path: str) -> List[Document]:
    """Load documents from a folder supporting PDF, DOCX, and TXT files."""
    documents = []
    
    if not os.path.exists(folder_path):
        raise ValueError(f"Folder path does not exist: {folder_path}")
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Skip directories
        if os.path.isdir(file_path):
            continue
            
        try:
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif filename.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            elif filename.endswith(".txt"):
                loader = TextLoader(file_path)
            else:
                print(f"Unsupported file type: {filename}")
                continue
            
            docs = loader.load()
            documents.extend(docs)
            print(f"Loaded {len(docs)} document(s) from {filename}")
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
            continue
    
    return documents


def create_vectorstore(
    documents: List[Document],
    collection_name: str = "my_collection",
    persist_directory: str = "./chroma_db",
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> Chroma:
    """Create and persist a Chroma vector store from documents."""
    # Initialize embedding function
    embedding_function = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    splits = text_splitter.split_documents(documents)
    print(f"Split the documents into {len(splits)} chunks.")
    
    # Create and persist vector store
    vectorstore = Chroma.from_documents(
        collection_name=collection_name,
        documents=splits,
        embedding=embedding_function,
        persist_directory=persist_directory
    )
    
    print(f"Vector store created and persisted to '{persist_directory}'")
    return vectorstore


def main():
    """Main execution function."""
    # Configuration
    folder_path = "sample_data"
    collection_name = "my_collection"
    persist_directory = "./chroma_db"
    
    # Load documents
    print(f"Loading documents from '{folder_path}'...")
    documents = load_documents(folder_path)
    print(f"Loaded {len(documents)} total documents from the folder.\n")
    
    if not documents:
        print("No documents found. Exiting.")
        return
    
    # Create vector store
    vectorstore = create_vectorstore(
        documents=documents,
        collection_name=collection_name,
        persist_directory=persist_directory
    )
    
    # Example query
    query = "How did the invention of agriculture influence the development of early human societies?"
    print(f"\nQuery: {query}")
    search_results = vectorstore.similarity_search(query, k=1)
    
    if search_results:
        print(f"\nTop result:\n{search_results[0].page_content}")
    else:
        print("No results found.")


if __name__ == "__main__":
    main()
