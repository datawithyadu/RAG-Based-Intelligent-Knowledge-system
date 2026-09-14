"""Select test sentences from the papers and save them to scripts/facts.json.

The facts are written to a file rather than printed for copying, because
hand-transcribing extracted PDF text loses characters - trailing spaces,
em dashes, and the spacing artifacts pypdf produces. A fact that differs
by one character can never be found, and the whole metric reads as failure.

Run this once. compare_chunking.py reads the file it produces.
"""

import json
import random
import re
from pathlib import Path

from app.services.ingest import extract_text

DATA_DIR = Path("data")
OUT_FILE = Path("scripts/facts.json")
MIN_LEN = 150               # long enough to span a chunk boundary
MAX_LEN = 300
PER_PAPER = 5               # how many facts to keep per paper
SKIP_FRONT = 0.08           # ignore the first 8% (cover, abstract, contents)
SKIP_BACK = 0.20            # ignore the last 20% (references, appendices)
SEED = 42                   # same sample every run, so results are reproducible


def is_claim(sentence: str) -> bool:
    """Reject anything that looks like a heading, citation or table row."""
    if not (MIN_LEN < len(sentence) < MAX_LEN):
        return False
    if sentence.count(".") > 3:                      # reference lists, ToC dots
        return False
    if len(re.findall(r"\d", sentence)) > 15:        # tables, page numbers
        return False
    if not sentence[0].isupper():                    # started mid-sentence
        return False
    if re.search(r"\[\d+\]|et al\.|https?://", sentence):
        return False
    return True


def candidates(text: str) -> list[str]:
    """Sentences from the body of the document that read like claims."""
    start = int(len(text) * SKIP_FRONT)
    end = int(len(text) * (1 - SKIP_BACK))
    body = text[start:end]

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body)]
    return [s for s in sentences if is_claim(s)]


random.seed(SEED)
facts: dict[str, list[str]] = {}

for pdf_path in sorted(DATA_DIR.glob("*.pdf")):
    try:
        text = extract_text(pdf_path)
    except Exception as e:
        print(f"{pdf_path.name} - FAILED: {e}")
        continue

    found = candidates(text)
    sample = random.sample(found, min(PER_PAPER, len(found)))

    # A sentence is only usable if it survives .strip() and is still in the
    # source text. Stripping can remove characters that were part of the match.
    verified = [s for s in sample if s in text]
    dropped = len(sample) - len(verified)

    facts[pdf_path.name] = verified

    print(f"{pdf_path.name:<28} {len(text):>9,} chars  "
          f"{len(found):>4} candidates  {len(verified)} kept"
          f"{f'  ({dropped} dropped)' if dropped else ''}")

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
OUT_FILE.write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")

total = sum(len(v) for v in facts.values())
print(f"\nWrote {total} facts to {OUT_FILE}")
print("Open it if you want to swap any sentence - but edit nothing inside the quotes.")
