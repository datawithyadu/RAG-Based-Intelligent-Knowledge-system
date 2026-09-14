"""Compare chunking strategies across the corpus.

Reads the test sentences written by scripts/pick_facts.py, so no fact is
ever hand-copied. Run pick_facts.py first.

Cheap strategies only by default. Set INCLUDE_SLOW = True to add
semantic and proposition, which cost API calls.
"""

import json
from pathlib import Path

from app.models import chunking as lab
from app.services.ingest import extract_text

DATA_DIR = Path("data")
FACTS_FILE = Path("scripts/facts.json")
INCLUDE_SLOW = False          # True adds semantic + proposition (costs money)
SAMPLE_CHARS = None           # e.g. 5000 to test on a slice first


if not FACTS_FILE.exists():
    raise SystemExit(f"{FACTS_FILE} not found - run: uv run python scripts/pick_facts.py")

FACTS: dict[str, list[str]] = json.loads(FACTS_FILE.read_text(encoding="utf-8"))


for pdf_path in sorted(DATA_DIR.glob("*.pdf")):
    facts = FACTS.get(pdf_path.name, [])

    text = extract_text(pdf_path)
    if SAMPLE_CHARS:
        text = text[:SAMPLE_CHARS]

    # Guard against a stale facts.json - if the facts were generated from a
    # different extraction they would score 0 everywhere and look like failure.
    missing = [f for f in facts if f not in text]
    if missing:
        print(f"\n!! {len(missing)} fact(s) not in {pdf_path.name} - re-run pick_facts.py")

    print(f"\n{'=' * 110}")
    print(f"{pdf_path.name}  -  {len(text):,} chars, {len(facts)} facts")
    print("=" * 110)

    result = lab.compare(text, facts=facts, include_slow=INCLUDE_SLOW)
    lab.print_table(result)