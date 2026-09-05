from pathlib import Path
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from boto3.exceptions import S3UploadFailedError
from app.config import Config

class StorageService:
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            aws_secret_access_key = Config.AWS_SECRET_ACCESS_KEY,
            aws_access_key_id = Config.AWS_ACCESS_KEY_ID,
            aws_region = Config.AWS_REGION
            )
        '''The above key will try to authentication with aws s3 bucket'''

        '''Now, let's initialize the bucket'''
        self.bucket = Config.AWS_BUCKET_NAME

        '''Upload method - My PDFs are in data/. 
        My project talks to AWS, uploads them to the S3 bucket, 
        and stores each one under a key I choose — which is the address I'll use to find it again.'''

    def upload_file(self,local_path:str, key: str):
         """Upload a file from disk. Returns the S3 key it was stored under."""
         try:
            self.s3_client.upload_file(local_path, self.bucket, key)
            return key
         except FileNotFoundError:
                print(f"File not found: {local_path}")
                return None
         except NoCredentialsError:
                print("AWS credentials not available.")
                return None
         except ClientError as e:
                print(f"Client error occurred: {e}")
                return None  
         except S3UploadFailedError as e:
              print(f"Upload faild{e}")
              return None                         

    def download_file(self, key: str, local_path: str):
        """Download one object from S3 to disk.

        Returns the local Path on success, or None if it failed.
        """
        destination = Path(local_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.s3_client.download_file(self.bucket, key, str(destination))
            return destination
        except NoCredentialsError:
            print("AWS credentials not available.")
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code in ("404", "NoSuchKey"):
                print(f"Not in bucket: {key}")
            else:
                print(f"Download failed ({code}): {key}")
        return None
             
