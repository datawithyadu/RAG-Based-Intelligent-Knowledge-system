"""S3 storage: move source documents between the bucket and local disk."""

from pathlib import Path

import boto3
from boto3.exceptions import S3UploadFailedError
from botocore.exceptions import ClientError, NoCredentialsError

from app.config import Config
from app.utils.logger import get_logger

logger = get_logger(__name__)


class S3Storage:
    """Reads and writes source documents in an S3 bucket."""

    def __init__(self):
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY,
            region_name=Config.AWS_REGION,
        )
        self.bucket = Config.AWS_BUCKET_NAME

    def upload_file(self, local_path: str, key: str) -> str | None:
        """Upload a file from disk. Returns the key on success, None on failure."""
        try:
            self.s3_client.upload_file(local_path, self.bucket, key)
            logger.info("Uploaded %s -> s3://%s/%s", local_path, self.bucket, key)
            return key
        except FileNotFoundError:
            logger.error("File not found: %s", local_path)
        except NoCredentialsError:
            logger.error("AWS credentials not available")
        except (S3UploadFailedError, ClientError) as e:
            logger.error("Upload of %s failed: %s", key, e)
        return None

    def download_file(self, key: str, local_path: str) -> Path | None:
        """Download one object to disk. Returns the local Path, or None on failure."""
        destination = Path(local_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.s3_client.download_file(self.bucket, key, str(destination))
            logger.info("Downloaded s3://%s/%s -> %s", self.bucket, key, destination)
            return destination
        except NoCredentialsError:
            logger.error("AWS credentials not available")
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code in ("404", "NoSuchKey"):
                logger.error("Not in bucket: %s", key)
            else:
                logger.error("Download of %s failed (%s)", key, code)
        return None

    def list_pdfs(self, prefix: str = "") -> list[str]:
        """Every PDF key in the bucket, following pagination."""
        keys: list[str] = []
        paginator = self.s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].lower().endswith(".pdf"):
                    keys.append(obj["Key"])
        logger.info("Found %s PDFs in s3://%s/%s", len(keys), self.bucket, prefix)
        return keys
