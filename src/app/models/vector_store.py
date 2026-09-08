"""Vector store: holds document chunks and finds the ones matching a question."""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Config


class VectorStore:
    """Wraps Chroma so the rest of the app never touches it directly."""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=Config.OPENAI_API_KEY,
        )

        self.db_path = Path(Config.VECTOR_DB_PATH)
        self.db_path.mkdir(parents=True, exist_ok=True)

        self.store = Chroma(
            collection_name="documents",
            embedding_function=self.embeddings,
            persist_directory=str(self.db_path),
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_CHILD,
            chunk_overlap=Config.CHUNK_OVERLAP,
        )

    def add_document(self, text: str, source: str) -> int:
        """Split one document into chunks and store them. Returns the chunk count."""
        chunks = self.splitter.split_text(text)

        documents = [
            Document(page_content=chunk, metadata={"source": source})
            for chunk in chunks
        ]

        self.store.add_documents(documents)
        return len(documents)

    def search(self, query: str, k: int | None = None) -> list[Document]:
        """Return the k chunks most similar to the query."""
        return self.store.similarity_search(query, k=k or Config.MAX_PARENTS)

    def count(self) -> int:
        """How many chunks are stored."""
        return len(self.store.get()["ids"])