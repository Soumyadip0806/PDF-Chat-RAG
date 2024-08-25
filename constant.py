import os
from chromadb.config import Settings

# ChromaDB configuration
CHROMADB_CONFIG = Settings(
    chroma_db_impl='duckdb+parquet',
    persist_directory="database/",
    anonymized_telemetry=False
)
