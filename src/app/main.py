"""FastAPI entry point for the RAG knowledge system."""

from contextlib import asynccontextmanager       # gives us @asynccontextmanager for lifespan
from pathlib import Path                          # safe, absolute file paths

from fastapi import FastAPI, HTTPException        # the framework + clean error responses
from pydantic import BaseModel                    # validates incoming request bodies

from app.models.vector_store import VectorStore   # our parent-child Chroma store
from app.services.llm_service import LLMService   # our answer generator
from app.utils.logger import get_logger           # our logging setup

"""Module level setup"""
logger = get_logger(__name__)
BASE_DIR = Path(__file__).resolve().parent        # folder that main.py lives in

store: VectorStore | None = None                  # filled in at startup
llm: LLMService | None = None                     # filled in at startup

"""Startup and shutdown"""
@asynccontextmanager
async def lifespan(app: FastAPI):
    global store, llm                             # assign to the module-level names, not new locals
    try:
        store = VectorStore()
        llm = LLMService()
        logger.info("Ready - %s chunks indexed", store.count())
    except Exception as e:
        logger.error("Startup failed: %s", e, exc_info=True)
        raise                                     # refuse to start in a broken state
    yield                                         # <-- server runs here
    logger.info("Shutting down")

"""The App Object"""
app = FastAPI(
title="RAG Knowledge System",
description="Ask questions against your indexed PDF library.",
version="0.1.0",
lifespan=lifespan,
)
"""Request and response shapes"""
class Question(BaseModel):
    text: str
    top_k: int = 3
class Source(BaseModel):
    source: str
    preview: str
class Answer(BaseModel):
    answer: str
    sources: list[Source]

"""Health endpoint"""
@app.get("/health")
def health():
    if store is None:
        raise HTTPException(status_code=503, detail="Store not initialised")
    return {"status": "ok", "chunks": store.count()}

"""Search endpoint"""
@app.post("/search")
def search(q: Question) -> list[Source]:
    if store is None:
        raise HTTPException(status_code=503, detail="Store not initialised")

    logger.info("Search: %s", q.text)
    try:
        docs = store.search(q.text)
    except Exception as e:
        logger.error("Search failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Search failed. Please try again.")

    return [
        Source(
            source=d.metadata.get("source", "unknown"),
            preview=d.page_content[:300],
        )
        for d in docs
    ]

"""Ask endpoint"""
@app.post("/ask")
def ask(q: Question) -> Answer:
    if store is None or llm is None:
        raise HTTPException(status_code=503, detail="Service not initialised")

    logger.info("Ask: %s", q.text)
    try:
        docs = store.search(q.text)
        text = llm.answer(q.text, docs)
    except Exception as e:
        logger.error("Ask failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Something went wrong while processing your request. Please try again.",
        )

    return Answer(
        answer=text,
        sources=[
            Source(
                source=d.metadata.get("source", "unknown"),
                preview=d.page_content[:300],
            )
            for d in docs
        ],
    )

def main():
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()