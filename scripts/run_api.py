#!/usr/bin/env python3
"""run_api.py — launches the DriveGuard AI FastAPI server via uvicorn."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import uvicorn

from src.config import API_HOST, API_PORT

if __name__ == "__main__":
    uvicorn.run("serving.api:app", host=API_HOST, port=API_PORT, reload=False)
