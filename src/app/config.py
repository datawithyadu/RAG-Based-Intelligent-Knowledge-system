"""All settings and secrets in one place."""

import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    # --- secrets ---
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
    AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
    AWS_BUCKET_NAME = os.environ["AWS_BUCKET_NAME"]
    AWS_REGION = os.getenv("AWS_REGION", "eu-central-1")

    # --- models ---
    EMBEDDING_MODEL = "text-embedding-3-small"
    CHAT_MODEL = "gpt-4o-mini"
    TEMPERATURE = 0.2

    # --- vector store ---
    VECTOR_DB_PATH = "vector_db"
    CHUNK_PARENT = 2000        # big chunk: what the LLM reads
    CHUNK_CHILD = 400          # small chunk: what gets searched
    CHUNK_OVERLAP = 100        # overlap between parents only
    TOP_K_CHILDREN = 8         # children retrieved per query
    MAX_PARENTS = 3            # parents sent to the LLM  <- the cost dial

    # --- logging ---
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_DIR = "logs"
    LOG_FILE = "app.log"
    LOG_MAX_BYTES = 1_000_000
    LOG_BACKUP_COUNT = 3
