#!/usr/bin/env python3
"""run_dashboard.py — launches the DriveGuard AI Streamlit dashboard."""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    app_path = Path(__file__).resolve().parent.parent / "frontend" / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)], check=True)
