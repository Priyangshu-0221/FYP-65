"""Application settings for the internship recommender backend."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    backend_host: str = Field("127.0.0.1", description="Host for the FastAPI backend")
    backend_port: int = Field(8000, description="Port for the FastAPI backend")
    resume_dataset_path: Path = Field(
        Path("DATA/resumes.csv"),
        description="Path to CSV or directory used to train the resume classifier.",
    )
    internship_catalog_path: Path | None = Field(
        default=None,
        description="Optional path to JSON file containing internship catalog overrides.",
    )

    class Config:
        env_prefix = "INTERNSHIP_APP_"
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
