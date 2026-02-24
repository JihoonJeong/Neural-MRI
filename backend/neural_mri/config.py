from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["http://localhost:5173"]

    # Model
    default_model: str = "gpt2"
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "mps"
    model_cache_dir: str | None = None

    # Cache
    max_cache_entries: int = 5  # LRU scan result cache size

    # Deployment
    environment: str = "local"  # "local" | "docker" | "huggingface"

    model_config = {"env_prefix": "NMRI_", "env_file": ".env"}
