"""Upload every PDF in data/ to S3.

Backfills documents that were indexed before the S3 wiring existed.
Safe to re-run - S3 overwrites by key, so nothing duplicates.
"""

from pathlib import Path

from app.services.storage_service import S3Storage

DATA_DIR = Path("data")
PREFIX = "documents"


s3 = S3Storage()
print(f"bucket: {s3.bucket}  region: {s3.s3_client.meta.region_name}")
print()

ok = failed = 0

for pdf in sorted(DATA_DIR.glob("*.pdf")):
    key = s3.upload_file(str(pdf), f"{PREFIX}/{pdf.name}")
    if key:
        ok += 1
    else:
        failed += 1
    print(f"  {pdf.name:<55} {key or 'FAILED'}")

print()
print(f"{ok} uploaded, {failed} failed")
