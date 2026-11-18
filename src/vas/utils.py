"""Small shared helpers: logging, table I/O, ffprobe."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

LOGGER_NAME = "vas"
_FMT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(level: str | int = "INFO") -> logging.Logger:
    """Configure the package logger once (console handler)."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level if isinstance(level, int) else level.upper())
    if not any(getattr(h, "_vas_console", False) for h in logger.handlers):
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(_FMT, datefmt="%H:%M:%S"))
        h._vas_console = True  # type: ignore[attr-defined]
        logger.addHandler(h)
    logger.propagate = False
    return logger


@contextmanager
def log_to_file(path: Path):
    """Temporarily also write package logs to ``path`` (per-session log)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter(_FMT))
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        handler.close()


def read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def write_table(df: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported table format: {path}")
    return path


def write_json(obj, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default))
    return path


def _json_default(o):
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serialisable: {type(o).__name__}")


def probe_fps(video: Path) -> float | None:
    """Return the average frame rate of ``video`` using ffprobe, or None."""
    if not shutil.which("ffprobe") or not Path(video).exists():
        return None
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate", "-of", "default=nw=1:nk=1", str(video),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
        num, _, den = out.partition("/")
        return float(num) / float(den or 1)
    except (subprocess.CalledProcessError, ValueError, ZeroDivisionError):
        return None
