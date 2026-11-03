"""
video_writer.py
===============
Writes the annotated frames (boxes, speeds, plates) of a processed video to an MP4 that **browsers can play**.

OpenCV's default ``mp4v`` codec is not playable in Chrome/Firefox, so frames are piped to the ffmpeg binary
that ships inside the ``imageio-ffmpeg`` wheel and encoded as H.264 (``yuv420p``, ``faststart``). If that
package is missing the writer falls back to OpenCV ``mp4v`` and sets ``browser_playable = False`` so the
caller can say so instead of showing a blank player.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from src.config import get_logger

logger = get_logger(__name__)


class AnnotatedVideoWriter:
    """Incrementally writes BGR frames to an MP4; the output size is fixed by the first frame."""

    def __init__(self, path: Path, fps: float) -> None:
        """Create the writer (the file is opened on the first frame).

        Args:
            path: Destination ``.mp4`` path; parent directories are created.
            fps: Playback frame rate of the output video.
        """
        self.path = Path(path)
        self.fps = max(1.0, float(fps))
        self.frames_written = 0
        self.browser_playable = True
        self._size: Optional[Tuple[int, int]] = None  # (width, height), both even
        self._ffmpeg = None
        self._cv_writer: Optional[cv2.VideoWriter] = None

    def _open(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        self._size = (w - w % 2, h - h % 2)  # yuv420p needs even dimensions
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import imageio_ffmpeg

            self._ffmpeg = imageio_ffmpeg.write_frames(
                str(self.path), self._size, fps=self.fps, codec="libx264", pix_fmt_in="bgr24", pix_fmt_out="yuv420p",
                macro_block_size=2, quality=6, ffmpeg_log_level="error",
                output_params=["-preset", "veryfast", "-movflags", "+faststart"],
            )
            self._ffmpeg.send(None)  # start the encoder
        except ImportError:
            logger.warning("imageio-ffmpeg is not installed: saving mp4v video that browsers may not play (pip install imageio-ffmpeg)")
            self.browser_playable = False
            self._cv_writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self._size)

    def write(self, frame: np.ndarray) -> None:
        """Append one BGR frame (cropped by at most one pixel to an even size)."""
        if self._size is None:
            self._open(frame)
        w, h = self._size  # type: ignore[misc]
        frame = np.ascontiguousarray(frame[:h, :w])
        if self._ffmpeg is not None:
            self._ffmpeg.send(frame.tobytes())
        elif self._cv_writer is not None:
            self._cv_writer.write(frame)
        self.frames_written += 1

    def close(self) -> Optional[Path]:
        """Finish the file. Returns its path, or None if no frame was ever written."""
        if self._ffmpeg is not None:
            self._ffmpeg.close()
            self._ffmpeg = None
        if self._cv_writer is not None:
            self._cv_writer.release()
            self._cv_writer = None
        return self.path if self.frames_written and self.path.exists() else None
