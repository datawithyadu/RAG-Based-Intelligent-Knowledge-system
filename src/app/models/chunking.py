"""Imports"""
"""Compare chunking strategies on the same text, side by side.

Not part of the running app. This is the measurement harness behind the
chunking decision recorded in the evaluation document.
"""

import json
import re
import statistics

from langchain_openai import ChatOpenAI
from langchain_text_splitters import (
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

from app.config import Config
from app.utils.logger import get_logger
logger = get_logger(__name__)

"""Fixed charactor splitter"""

def fixed_char(text: str, size: int = 400, overlap: int = 0) -> list[str]:
    """Cut every N characters. Ignores sentence and word boundaries."""
    splitter = CharacterTextSplitter(
        separator="",
        chunk_size=size,
        chunk_overlap=overlap,
    )
    return splitter.split_text(text)

"""Token based """
def token_based(text: str, size: int = 100, overlap: int = 10) -> list[str]:
    """Chunk by model tokens rather than characters."""
    splitter = TokenTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
    )
    return splitter.split_text(text)

"""Sliding window approach"""
def sliding_window(text: str, size: int = 400, overlap: int = 200) -> list[str]:
    """Fixed chunks with heavy overlap. Nothing falls between two chunks -
    at the cost of storing much of the text more than once."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,       
    )
    return splitter.split_text(text)

"""Recursive Charater split"""
def recursive(text: str, size: int = 400, overlap: int = 50) -> list[str]:
    """Try to break on paragraph, then sentence, then word, then character."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
    )
    return splitter.split_text(text)

"""Structure aware"""
def structure_aware(text: str) -> list[str]:
    """Split on markdown headings. Only useful if the text HAS headings."""
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
    )
    docs = splitter.split_text(text)

    if len(docs) <= 1:
        logger.warning("structure_aware: no markdown headings found - returned 1 chunk")

    return [d.page_content for d in docs]

"""Semantic"""
def semantic(text: str) -> list[str]:
    """Embed sentences, cut where consecutive sentences stop being similar.
    """
    try:
        from langchain_experimental.text_splitter import SemanticChunker
        from langchain_openai import OpenAIEmbeddings
    except ImportError:
        logger.warning("semantic: langchain-experimental not installed - skipped")
        return []

    splitter = SemanticChunker(
        OpenAIEmbeddings(
            model=Config.EMBEDDING_MODEL,
            api_key=Config.OPENAI_API_KEY,
        )
    )
    return splitter.split_text(text)

"""Parent Child"""

def parent_child(text: str) -> dict[str, list[str]]:
    """Big parents for context, small children for matching."""
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_PARENT,
        chunk_overlap=Config.CHUNK_OVERLAP,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_CHILD,
        chunk_overlap=0,
    )

    parents = parent_splitter.split_text(text)

    children = []
    for parent in parents:
        children.extend(child_splitter.split_text(parent))

    return {"parents": parents, "children": children}

"""Porposition"""
PROPOSITION_PROMPT = """Break the passage below into a list of standalone factual statements.

Rules:
- Each statement must be understandable on its own, with no outside context.
- Replace every pronoun and reference ("it", "this", "the authors", "the model")
  with the actual name it refers to.
- Split any sentence that contains more than one fact.
- Use ONLY information present in the passage. Do not add, infer or explain.
- If the passage is a table of contents, a page header, a reference list or
  otherwise contains no factual claims, return an empty list.

Return a JSON array of strings and nothing else.

Passage:
\"\"\"{passage}\"\"\"
"""


def _parse_propositions(raw: str) -> list[str]:
    """Pull a JSON list out of the model's reply, tolerating code fences."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("proposition: reply was not valid JSON - falling back to lines")
        return [ln.strip("-* ").strip() for ln in cleaned.splitlines() if ln.strip()]

    if not isinstance(parsed, list):
        return []

    return [str(p).strip() for p in parsed if str(p).strip()]


def proposition(text: str, passage_size: int = 1000) -> list[str]:
    """Decompose the text into atomic facts using one LLM call per passage."""
    llm = ChatOpenAI(
        model=Config.CHAT_MODEL,
        temperature=0,                      # decomposition must not be creative
        api_key=Config.OPENAI_API_KEY,
    )

    passages = RecursiveCharacterTextSplitter(
        chunk_size=passage_size,
        chunk_overlap=0,
    ).split_text(text)

    logger.info("proposition: decomposing %s passages", len(passages))

    props: list[str] = []
    for i, passage in enumerate(passages, 1):
        try:
            reply = llm.invoke(PROPOSITION_PROMPT.format(passage=passage)).content
            props.extend(_parse_propositions(reply))
        except Exception as e:
            logger.error("proposition: passage %s failed: %s", i, e)

    return props

"""Measurement"""
def stats(chunks: list[str]) -> dict:
    """Shape of a chunk list: how many, how big, how even."""
    if not chunks:
        return {"count": 0, "mean": 0, "min": 0, "max": 0, "stdev": 0}

    sizes = [len(c) for c in chunks]
    return {
        "count": len(chunks),
        "mean": round(statistics.mean(sizes)),
        "min": min(sizes),
        "max": max(sizes),
        "stdev": round(statistics.stdev(sizes)) if len(sizes) > 1 else 0,
    }


def intact_facts(chunks: list[str], facts: list[str]) -> dict[str, bool]:
    """Did each fact survive in ONE chunk, or did a split cut it in half?"""
    return {fact: any(fact in c for c in chunks) for fact in facts}


def compare(text: str, facts: list[str] | None = None, include_slow: bool = False) -> dict:
    """Run every strategy on the same text. include_slow adds the API-cost ones."""
    results: dict[str, list[str]] = {
        "fixed_char": fixed_char(text),
        "sliding_window": sliding_window(text),      # <-- the only new line
        "recursive": recursive(text),
        "token_based": token_based(text),
        "structure_aware": structure_aware(text),
    }

    pc = parent_child(text)
    results["parent_child (children)"] = pc["children"]
    results["parent_child (parents)"] = pc["parents"]

    if include_slow:
        results["semantic"] = semantic(text)
        results["proposition"] = proposition(text)

    table = {name: stats(chunks) for name, chunks in results.items()}

    if facts:
        for name, chunks in results.items():
            survived = intact_facts(chunks, facts)
            table[name]["facts_intact"] = f"{sum(survived.values())}/{len(facts)}"

    return {"stats": table, "chunks": results}


def print_table(comparison: dict) -> None:
    """Print the comparison as an aligned table."""
    table = comparison["stats"]

    cols = ["count", "mean", "min", "max", "stdev"]
    has_facts = any("facts_intact" in row for row in table.values())
    if has_facts:
        cols.append("facts_intact")

    header = f"{'strategy':<26}" + "".join(f"{c:>14}" for c in cols)
    print(header)
    print("-" * len(header))

    for name, row in table.items():
        line = f"{name:<26}" + "".join(f"{str(row.get(c, '-')):>14}" for c in cols)
        print(line)