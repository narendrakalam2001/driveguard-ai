"""Download model weights from the HuggingFace Hub into ``models/`` (used at container start-up).

Weights are not baked into the Docker image or committed to git (see .gitignore); a deploy target
instead fetches them once.

Usage:
    HF_MODEL_REPO=your-user/driveguard-ai-models python scripts/download_models.py

Missing repo/files are logged and skipped: the pipeline then falls back to base COCO weights, so the
container still starts.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import HUGGINGFACE_TOKEN, MODELS_DIR, get_logger

logger = get_logger("download_models")
FILES = ["yolov8_vehicle.pt", "yolov8_plate.pt"]


def main() -> int:
    """Fetch each weight file that is not already present; return the number of failures."""
    repo = os.getenv("HF_MODEL_REPO", "")
    if not repo or repo.startswith("your-"):
        logger.warning("HF_MODEL_REPO is not set; skipping model download (base COCO weights will be used).")
        return 0
    from huggingface_hub import hf_hub_download

    failures = 0
    for name in FILES:
        if (MODELS_DIR / name).exists():
            logger.info("%s already present", name)
            continue
        try:
            hf_hub_download(repo_id=repo, filename=name, local_dir=str(MODELS_DIR), token=HUGGINGFACE_TOKEN or None)
            logger.info("Downloaded %s", name)
        except Exception as exc:  # noqa: BLE001 - never block start-up on a model fetch
            failures += 1
            logger.error("Could not download %s from %s: %s", name, repo, exc)
    return failures


if __name__ == "__main__":
    main()
    sys.exit(0)
