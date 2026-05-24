from __future__ import annotations

import os
import secrets

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

SECRET_KEY: str = os.getenv("SECRET_KEY", secrets.token_hex(32))
ALGORITHM: str = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
REFRESH_TOKEN_EXPIRE_DAYS: int = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./backend.db")

# Allowed origins for CORS (comma-separated in env)
_origins_env = os.getenv("CORS_ORIGINS", "http://localhost:8501")
CORS_ORIGINS: list[str] = [o.strip() for o in _origins_env.split(",")]
