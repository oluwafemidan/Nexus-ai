from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()
# Nexus knowledge base configuration
KNOWLEDGE_FOLDER = Path("knowledge_base")
VECTOR_FOLDER = "./vector_store"
COLLECTION_NAME = "nexus_company_knowledge"

# Embedding model

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

def build_knowledge_base():

    if not KNOWLEDGE_FOLDER.exists():
        raise FileNotFoundError(
            "Nexus knowledge folder not found."
        )

    documents = []

    # Automatically read all TXT documents
    for file_path in KNOWLEDGE_FOLDER.glob("*.txt"):

        content = file_path.read_text(
            encoding="utf-8"
        )

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": file_path.name,
                    "company": "Nexus Workplace Services"
                }
            )
        )

    if not documents:
        raise ValueError(
            "No Nexus policy documents were found."
        )

    # Split documents into overlapping chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")

    # Create a fresh vector collection
    database = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=VECTOR_FOLDER
    )

    # Clear old indexed content before rebuilding
    database.reset_collection()

    database.add_documents(chunks)

    print("Nexus knowledge base indexed successfully.")
    print(f"Total indexed chunks: {len(chunks)}")


if __name__ == "__main__":
    build_knowledge_base()
