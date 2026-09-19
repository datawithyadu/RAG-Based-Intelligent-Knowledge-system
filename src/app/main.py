"""FastAPI entry point for the RAG knowledge system."""

from contextlib import asynccontextmanager       # gives us @asynccontextmanager for lifespan
from pathlib import Path                          # safe, absolute file paths

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse        # serves a single file back to the browser
from fastapi.staticfiles import StaticFiles       # serves a whole folder (css, js, images)
from pydantic import BaseModel                    # validates incoming request bodies

from app.models.vector_store import VectorStore   # our parent-child Chroma store
from app.services.ingest import IngestError, process_document
from app.services.llm_service import LLMService   # our answer generator
from app.services.storage_service import S3Storage
from app.utils.logger import get_logger           # our logging setup

# ---- Module level setup ----
logger = get_logger(__name__)
BASE_DIR = Path(__file__).resolve().parent        # folder that main.py lives in
PROJECT_ROOT = BASE_DIR.parent.parent             # src/app -> src -> project root
STATIC_DIR = BASE_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"
DATA_DIR = PROJECT_ROOT / "data"

MAX_UPLOAD_BYTES = 30 * 1024 * 1024               # 30 MB - your papers are 5-8 MB
MAX_TOTAL_PARENTS = 6                             # ceiling on merged context across sub-questions

store: VectorStore | None = None                  # filled in at startup
llm: LLMService | None = None                     # filled in at startup
s3: S3Storage | None = None                       # None if AWS is unreachable


# ---- Startup and shutdown ----
@asynccontextmanager
async def lifespan(app: FastAPI):
    global store, llm, s3
    try:
        store = VectorStore()
        llm = LLMService()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Ready - %s chunks indexed", store.count())
    except Exception as e:
        logger.error("Startup failed: %s", e, exc_info=True)
        raise                                     # refuse to start in a broken state

    # S3 is optional. A bad key or an offline bucket should not stop the
    # app from answering questions about documents already indexed.
    try:
        s3 = S3Storage()
        logger.info("S3 ready - bucket %s", s3.bucket)
    except Exception as e:
        s3 = None
        logger.warning("S3 unavailable, uploads stay local only: %s", e)

    yield                                         # <-- server runs here
    logger.info("Shutting down")


# ---- The app object ----
app = FastAPI(
    title="RAG Knowledge System",
    description="Ask questions against your indexed PDF library.",
    version="0.1.0",
    lifespan=lifespan,
)


# ---- Request and response shapes ----
class Question(BaseModel):
    text: str
    top_k: int = 3


class Source(BaseModel):
    source: str
    preview: str


class Answer(BaseModel):
    answer: str
    sources: list[Source]


class UploadResult(BaseModel):
    filename: str
    parents: int
    chunks: int                                   # total in the store afterwards
    s3_key: str | None = None                     # None if S3 was unavailable


def to_sources(docs) -> list[Source]:
    """Turn retrieved documents into the shape the API returns."""
    return [
        Source(
            source=d.metadata.get("source", "unknown"),
            preview=d.page_content[:300],
        )
        for d in docs
    ]


# ---- Health endpoint ----
@app.get("/health")
def health():
    if store is None:
        raise HTTPException(status_code=503, detail="Store not initialised")
    return {"status": "ok", "chunks": store.count()}


# ---- Search endpoint: retrieval only, no LLM ----
@app.post("/search")
def search(q: Question) -> list[Source]:
    if store is None:
        raise HTTPException(status_code=503, detail="Store not initialised")

    logger.info("Search: %s", q.text)
    try:
        docs = store.search(q.text, k=q.top_k)
    except Exception as e:
        logger.error("Search failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Search failed. Please try again.")

    return to_sources(docs)


# ---- Ask endpoint: retrieval + LLM ----
@app.post("/ask")
def ask(q: Question) -> Answer:
    if store is None or llm is None:
        raise HTTPException(status_code=503, detail="Service not initialised")

    logger.info("Ask: %s", q.text)
    try:
        # 1. Split the message into the questions it actually contains.
        subs = llm.decompose(q.text)

        # 2. Retrieve separately for each, so each gets its own nearest
        #    neighbours instead of one blended query vector.
        merged: list = []
        seen: set[str] = set()

        for sub in subs:
            for doc in store.search(sub, k=q.top_k):
                pid = doc.metadata.get("parent_id")
                if pid in seen:                   # same parent found twice
                    continue
                seen.add(pid)
                merged.append(doc)

        # 3. Cap the context so a four-part question can't blow the budget.
        if len(merged) > MAX_TOTAL_PARENTS:
            logger.info("Trimming %s parents to %s", len(merged), MAX_TOTAL_PARENTS)
            merged = merged[:MAX_TOTAL_PARENTS]

        # 4. One answer, over the union of everything retrieved.
        text = llm.answer(q.text, merged)

    except Exception as e:
        logger.error("Ask failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Something went wrong while processing your request. Please try again.",
        )

    return Answer(answer=text, sources=to_sources(merged))


# ---- Upload endpoint: save a PDF, extract it, add it to the index ----
@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> UploadResult:
    if store is None:
        raise HTTPException(status_code=503, detail="Store not initialised")

    # Take only the final component: a filename like "../../.env" must not
    # be able to write outside DATA_DIR.
    name = Path(file.filename or "").name
    if not name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    dest = DATA_DIR / name
    if dest.exists():
        raise HTTPException(status_code=409, detail=f"{name} is already in the library")

    # Stream to disk in 1 MB pieces rather than loading the whole file into
    # memory, and stop if it exceeds the cap.
    written = 0
    try:
        with dest.open("wb") as out:
            while piece := await file.read(1024 * 1024):
                written += len(piece)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File is larger than {MAX_UPLOAD_BYTES // 1024 // 1024} MB",
                    )
                out.write(piece)
    except HTTPException:
        dest.unlink(missing_ok=True)              # don't leave a partial file behind
        raise
    except Exception as e:
        dest.unlink(missing_ok=True)
        logger.error("Upload write failed for %s: %s", name, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Could not save the file")

    logger.info("Uploaded %s (%s bytes)", name, written)

    # Index it. A PDF that yields no text is a user-fixable problem (scanned
    # image), so it gets a 422 with a real explanation, not a generic 500.
    try:
        parents = process_document(dest, store)
    except IngestError as e:
        dest.unlink(missing_ok=True)
        logger.warning("Ingest rejected %s: %s", name, e)
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        dest.unlink(missing_ok=True)
        logger.error("Ingest failed for %s: %s", name, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Could not index the document")

    # Archive to S3 only after indexing succeeded. If ingest rejects the
    # file we have already deleted it, and a rejected document should not
    # end up in the bucket.
    s3_key = None
    if s3 is not None:
        s3_key = s3.upload_file(str(dest), f"documents/{name}")
        if s3_key is None:
            logger.warning("S3 upload failed for %s - local copy only", name)

    return UploadResult(
        filename=name,
        parents=parents,
        chunks=store.count(),
        s3_key=s3_key,
    )


# ---- Web page ----
# Registered only if the file exists, so the API still boots without it.
if INDEX_FILE.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def read_index():
        return FileResponse(INDEX_FILE)
else:
    logger.warning("No %s - browse the API at /docs instead", INDEX_FILE)


def main():
    import uvicorn

    # Watch only source code. Without reload_dirs the watcher sees every
    # write into data/ and vector_db/ as a code change, so uploading a PDF
    # restarts the server in the middle of indexing it.
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_dirs=[str(BASE_DIR.parent)],       # the src/ folder
    )


if __name__ == "__main__":
    main()