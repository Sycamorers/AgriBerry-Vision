from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, TypeVar

T = TypeVar("T")


def setup_logger(name: str = "detection_distribution", level: int = logging.INFO) -> logging.Logger:
    """Create a consistent console logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def ensure_dir(path: Path | str) -> Path:
    """Create directory (including parents) and return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def select_weights(weights: str | None, ckpt_dir: Path | str = "ckpt") -> Path:
    """Pick YOLO weights according to priority rules.

    Priority when ``weights`` is not explicitly provided:
    1) ckpt/best.pt
    2) ckpt/last.pt
    3) newest-modified *.pt under ckpt/
    """
    if weights:
        candidate = Path(weights)
        if not candidate.exists():
            raise FileNotFoundError(f"Weights file not found: {candidate}")
        return candidate

    ckpt_path = Path(ckpt_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {ckpt_path}")

    best = ckpt_path / "best.pt"
    if best.exists():
        return best

    last = ckpt_path / "last.pt"
    if last.exists():
        return last

    candidates = [p for p in ckpt_path.rglob("*.pt") if p.is_file()]
    if not candidates:
        raise FileNotFoundError(
            f"No .pt weights found under {ckpt_path}. Provide --weights explicitly."
        )

    return max(candidates, key=lambda p: p.stat().st_mtime)


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_class_name(name: str) -> str:
    """Make class name safe for file names."""
    safe = _SAFE_NAME_RE.sub("_", str(name).strip())
    safe = safe.strip("._")
    return safe or "class"


def chunked(items: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    """Yield fixed-size chunks from a sequence."""
    if size <= 0:
        raise ValueError("Chunk size must be > 0")
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    """Remove duplicates while preserving original order."""
    seen = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out
