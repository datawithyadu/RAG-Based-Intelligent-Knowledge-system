"""Index every PDF in data/ and run one search to confirm retrieval works."""

from app.models.vector_store import VectorStore
from app.services.ingest import process_folder

store = VectorStore()

print("=== indexing ===")
results = process_folder("data", store)
for name, outcome in results.items():
    print(f"  {name:<32} -> {outcome}")

print(f"\ntotal children indexed: {store.count()}")

print("\n=== search ===")
for doc in store.search("what is agentic AI?"):
    print(f"\n[{doc.metadata['source']}]")
    print(doc.page_content[:300])
