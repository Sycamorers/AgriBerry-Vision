from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class SequenceInfo:
    box: str
    seq: str
    seq_dir: Path


def discover_sequences(
    data_dir: Path | str,
    boxes: Sequence[str] = ("box_1", "box_2"),
) -> List[SequenceInfo]:
    """Discover immediate sequence folders under data/<box>/*."""
    data_root = Path(data_dir)
    all_sequences: List[SequenceInfo] = []

    for box in boxes:
        box_dir = data_root / box
        if not box_dir.exists() or not box_dir.is_dir():
            continue

        for seq_dir in sorted((p for p in box_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
            all_sequences.append(SequenceInfo(box=box, seq=seq_dir.name, seq_dir=seq_dir))

    return all_sequences


def collect_image_files(
    sequence_dir: Path | str,
    image_extensions: Iterable[str] = IMAGE_EXTENSIONS,
) -> List[Path]:
    """Recursively collect all image files under a sequence directory."""
    seq_dir = Path(sequence_dir)
    ext_set = {e.lower() for e in image_extensions}

    files = [
        p
        for p in seq_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in ext_set
    ]
    files.sort(key=lambda p: str(p))
    return files
