'''Load all the secret keys here'''
import os 
from dotenv import load_dotenv
load_dotenv()
class Config:
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
    AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
    AWS_BUCKET_NAME = os.environ["AWS_BUCKET_NAME"]
    AWS_REGION = os.getenv("AWS_REGION", "eu-central-1")
    VECTOR_DB_PATH = "vector_db"
