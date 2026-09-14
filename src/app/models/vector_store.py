"""Parent-child vector store: search small chunks, return the large ones."""

import json
import uuid
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Config
from app.utils.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    """Embeds small child chunks; returns the big parent chunk each came from."""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=Config.OPENAI_API_KEY,
        )

        self.db_path = Path(Config.VECTOR_DB_PATH)
        self.db_path.mkdir(parents=True, exist_ok=True)

        self.store = Chroma(
            collection_name="children",
            embedding_function=self.embeddings,
            persist_directory=str(self.db_path),
        )

        self.parents_file = self.db_path / "parents.json"
        self.parents = self._load_parents()

        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_PARENT,
            chunk_overlap=Config.CHUNK_OVERLAP,
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_CHILD,
            chunk_overlap=0,
        )

    def _load_parents(self) -> dict:
        if self.parents_file.exists():
            return json.loads(self.parents_file.read_text(encoding="utf-8"))
        return {}

    def _save_parents(self) -> None:
        self.parents_file.write_text(
            json.dumps(self.parents, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def add_document(self, text: str, source: str) -> int:
        """Split into parents, then children. Embeds only the children."""
        parent_chunks = self.parent_splitter.split_text(text)

        children = []
        for parent_text in parent_chunks:
            parent_id = str(uuid.uuid4())
            self.parents[parent_id] = {"text": parent_text, "source": source}

            for child_text in self.child_splitter.split_text(parent_text):
                children.append(Document(
                    page_content=child_text,
                    metadata={"parent_id": parent_id, "source": source},
                ))

        self.store.add_documents(children)
        self._save_parents()
        logger.info(
            "Indexed %s: %s parents, %s children",
            source, len(parent_chunks), len(children),
        )
        return len(parent_chunks)

    def search(self, query: str, k: int | None = None) -> list[Document]:
        """Search children, return the unique parents they belong to."""
        max_parents = k or Config.MAX_PARENTS

        hits = self.store.similarity_search(query, k=Config.TOP_K_CHILDREN)

        seen = []
        for hit in hits:
            parent_id = hit.metadata["parent_id"]
            if parent_id not in seen:
                seen.append(parent_id)              # keeps best-match order
            if len(seen) >= max_parents:
                break

        logger.info("%s children matched -> %s unique parents", len(hits), len(seen))

        return [
            Document(
                page_content=self.parents[pid]["text"],
                metadata={"source": self.parents[pid]["source"], "parent_id": pid},
            )
            for pid in seen
        ]

    def count(self) -> int:
        """Number of child chunks stored."""
        return len(self.store.get()["ids"])