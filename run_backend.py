#!/usr/bin/env python
"""Start the FastAPI backend.

Usage:
    python run_backend.py              # default: port 8000, auto-reload off
    python run_backend.py --reload     # dev mode with auto-reload
    python run_backend.py --port 9000
"""
from __future__ import annotations

import argparse

import uvicorn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run(
        "backend.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
