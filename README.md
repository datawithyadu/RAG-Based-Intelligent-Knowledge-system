# RAG-Based-Intelligent-Knowledge-system

<img width="1722" height="1058" alt="image" src="https://github.com/user-attachments/assets/ba61e76f-e8c0-4714-8e5e-116167a7e767" />
# RAG-Based Intelligent Knowledge System

Ask questions against a library of PDFs and get answers grounded in the
source text, with the passages they came from.

Built to test one question properly: **does chunking strategy actually
matter, and by how much?** The answer, measured on a real corpus, is in
[Measured results](#measured-results).

---

## What it does

Upload a PDF through the browser or the API. The system extracts the text,
splits it into a two-level parent–child structure, embeds the small chunks,
and stores them in Chroma. A question is decomposed into its sub-questions,
each is retrieved for separately, and the results are merged and answered by
an LLM — with every source passage returned alongside the answer.

Source documents are archived to S3 after indexing.

---

## Architecture

### Parent–child retrieval

The core design decision. Two chunk sizes, used for different jobs:

```
document
  └── parent chunk (2000 chars)   ← returned to the LLM
        ├── child chunk (400)     ← embedded and searched
        ├── child chunk (400)
        └── child chunk (400)
```

Only children are embedded. A search finds the 8 nearest children, maps each
back to its parent, deduplicates, and returns up to 3 unique parents.

The reason: small chunks match precisely — a 400-character chunk is about one
idea, so its embedding is sharp. But 400 characters is too little to answer
from; it cuts off mid-argument. Large chunks have the opposite problem — enough
context to answer, but a vector that averages several topics and matches
nothing well.

Parent–child gets both. Search on the fragment, answer from the passage.

### Query decomposition

`"What does paper A conclude, and what is X?"` embedded as one string produces
a vector sitting between two topics, matching neither. So the message is first
split into standalone sub-questions, each retrieved for independently, and the
deduplicated union is passed to the LLM as one context.

Costs one extra LLM call per request. Every failure path falls back to treating
the message as a single question.

### Layers

| Layer | File | Responsibility |
|---|---|---|
| API | `src/app/main.py` | endpoints, validation, error shape |
| Retrieval | `src/app/models/vector_store.py` | chunking, embedding, parent–child search |
| Generation | `src/app/services/llm_service.py` | decomposition, prompting |
| Ingestion | `src/app/services/ingest.py` | PDF → text → store |
| Archival | `src/app/services/storage_service.py` | S3 upload/download/list |
| Config | `src/app/config.py` | every setting and secret, one file |

Endpoints stay thin. All mechanics live in services.

---

## Measured results

Seven chunking strategies, run on the corpus with
`scripts/compare_chunking.py`. Facts-intact counts whether a known sentence
from the paper survives whole inside one chunk — a fact split across two
chunks can never be retrieved intact, whatever the retriever does.

| Strategy | Facts intact | Chunks (avg) |
|---|---|---|
| **parent–child (parents)** | **15/15** | **83** |
| sliding window | 15/15 | 802 |
| structure-aware | 10/15 | 1 |
| parent–child (children) | 9/15 | 494 |
| token-based | 8/15 | 444 |
| fixed character | 6/15 | 390 |
| recursive | 4/15 | 438 |

**Findings**

1. **Structure-aware chunking is inapplicable to PDF input.** Extracted PDF
   text has no markdown headings, so it found no boundaries and returned each
   document whole. It also silently dropped ~1% of the input text.
2. **Fixed-character splitting destroys facts** — 6/15, cutting mid-word by
   design.
3. **Parent–child matches sliding window's retention at ~10% of the chunk
   count.** Embedding cost scales with chunk count, so that is roughly a 10×
   difference in index cost for equal retention.
4. **The two parent–child rows are the argument.** Children: 494 small chunks,
   9/15 — precise to search, too fragmented to answer from. Parents: 83 large
   chunks, 15/15 — complete enough to answer from, too coarse to search. No
   single-level strategy has both properties.

**Caveats, stated rather than buried**

- Parents are ~1,950 characters and sliding-window chunks ~360. A larger chunk
  retains more facts almost by definition, so finding 3 is partly restating
  "bigger pieces". The defensible claim is finding 4.
- `recursive` scored *below* `fixed_char`. At n=15 that is within noise and
  should not be reported as a result.
- These numbers hold for long academic PDFs queried conceptually. For short
  documents or lookup questions, smaller focused chunks would likely win.
  There is no corpus-independent best strategy.
- Fact retention is a proxy for the *ceiling* on recall, not recall itself.
  It rewards never splitting — which is why structure-aware scores 10/15 while
  being unusable. Read it alongside the chunk-count column or not at all.

Test sentences are generated by `scripts/pick_facts.py` into `facts.json`
rather than copied by hand: extracted PDF text contains spacing artifacts that
do not survive manual transcription, and a fact differing by one character is
never found.

---

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/datawithyadu/RAG-Based-Intelligent-Knowledge-system
cd RAG-Based-Intelligent-Knowledge-system
uv sync
```

Create `.env` in the project root:

```
OPENAI_API_KEY=sk-...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_BUCKET_NAME=your-bucket
AWS_REGION=us-east-1
```

No quotes around values — `dotenv` treats them as part of the string.

Put PDFs in `data/`, then index them:

```bash
uv run python scripts/ingest_all.py
```

Run the server:

```bash
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000> for the web UI, or `/docs` for the API.

### AWS permissions

The IAM user needs an inline policy scoped to the one bucket:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Effect": "Allow", "Action": "s3:ListBucket",
      "Resource": "arn:aws:s3:::your-bucket" },
    { "Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject"],
      "Resource": "arn:aws:s3:::your-bucket/*" }
  ]
}
```

`ListBucket` acts on the bucket ARN; object actions act on `bucket/*`. They
cannot share one statement.

---

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | liveness, plus the number of indexed chunks |
| `POST` | `/search` | retrieval only, no LLM — for debugging |
| `POST` | `/ask` | retrieval + generation, returns answer and sources |
| `POST` | `/upload` | accept a PDF, index it, archive it to S3 |

`/search` exists deliberately. When an answer is wrong there are three possible
causes, and they are checked in order: is it indexed (`/health`), does it come
back (`/search`), does the LLM use it (`/ask`). Without a retrieval-only
endpoint, diagnosing that is guesswork.

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"text": "What are the main challenges in multi-agent coordination?", "top_k": 3}'
```

---

## Configuration

Every tunable lives in `src/app/config.py`:

| Setting | Default | Notes |
|---|---|---|
| `CHUNK_PARENT` | 2000 | characters, not tokens |
| `CHUNK_CHILD` | 400 | |
| `CHUNK_OVERLAP` | 100 | parents only; children use 0 |
| `TOP_K_CHILDREN` | 8 | children retrieved per query |
| `MAX_PARENTS` | 3 | unique parents returned |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | |
| `CHAT_MODEL` | `gpt-4o-mini` | |

Child overlap is 0 deliberately. Measured separately: at child size 400,
overlap changes neither fact coverage nor chunk count, and the parent is the
safety net — facts survive whole in a parent at every setting tested. Below
~200 characters overlap becomes necessary.

---

## Project structure

```
src/app/
  main.py                 FastAPI app and endpoints
  config.py               all settings and secrets
  models/
    vector_store.py       parent-child Chroma store
    chunking.py           seven strategies, for comparison only
  services/
    ingest.py             PDF -> text -> store
    llm_service.py        decomposition and answering
    storage_service.py    S3
  utils/logger.py         console + rotating file
  static/index.html       web UI
scripts/
  ingest_all.py           index everything in data/
  compare_chunking.py     run the chunking comparison
  pick_facts.py           generate test sentences -> facts.json
  push_to_s3.py           backfill existing documents to S3
```

`models/chunking.py` is a measurement harness, not part of the running app.
Nothing in `main.py` imports it.

---

## Not done

Listed because a portfolio project that claims to be finished invites the
question of what was skipped.

- **Retrieval evaluation.** Nothing here measures whether the *correct* chunk
  comes back for a real question. That needs a tagged question set and
  recall@k, and it is the number that would actually justify the architecture.
- **Front-matter filtering.** A table-of-contents fragment has been observed
  occupying one of three returned parents on a live query — a third of the
  context window carrying no information.
- **Hybrid search.** Vector-only retrieval misses exact terms. A query using a
  name for a document that does not appear inside it retrieves nothing useful.
- **S3 as storage of record.** Documents are archived to the bucket, but
  nothing reads from it. A fresh machine cannot rebuild the library from S3.
- **Semantic and proposition chunking rows** in the comparison — both need API
  calls and are gated behind a flag.
- **Tests.** There are none.

---

## Licence

MIT
