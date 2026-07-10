"""
NeuroAssist AI v2 — Application Configuration
Loads all settings from environment variables via pydantic-settings.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List
import os


class Settings(BaseSettings):
    """Central configuration loaded from .env file."""

    # ---- App ----
    app_name: str = "NeuroAssist AI"
    app_version: str = "2.0.0"
    debug: bool = False

    # ---- MongoDB ----
    mongodb_uri: str = "mongodb://localhost:27017/neuroassist_v2"
    mongodb_db_name: str = "neuroassist_v2"

    # ---- JWT ----
    jwt_secret_key: str = Field(..., min_length=32)
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 480
    jwt_refresh_token_expire_days: int = 7

    # ---- Gemini (ONLY LLM provider) ----
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    # ---- Server ----
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # ---- Storage ----
    storage_backend: str = "local"
    local_upload_dir: str = "./uploads/mri"
    file_encryption_key: str = ""

    # ---- ML ----
    model_save_dir: str = "./ml/checkpoints"
    best_model_path: str = "./ml/checkpoints/best_model.pt"

    # ---- RAG ----
    rag_docs_dir: str = "./backend/services/rag/guideline_docs"
    faiss_index_dir: str = "./backend/services/rag/faiss_index"
    embedding_model: str = "all-MiniLM-L6-v2"
    rag_top_k: int = 5

    # ---- PDF ----
    pdf_output_dir: str = "./exports/pdfs"

    # ---- Redis ----
    redis_url: str = "redis://localhost:6379/0"

    # ---- Compliance ----
    decision_support_mode_only: bool = True

    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.cors_origins.split(",")]

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


# Singleton instance
settings = Settings()
