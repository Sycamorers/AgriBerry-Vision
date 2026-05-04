from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrialFolder:
    date: str
    box: str
    trial: str
    trial_dir: Path

    @property
    def run_id(self) -> str:
        return f"{self.date}/{self.box}/{self.trial}"

    @property
    def output_rel_path(self) -> Path:
        return Path(self.date) / self.box / self.trial


def sorted_child_dirs(path: Path) -> list[Path]:
    if not path.exists() or not path.is_dir():
        return []
    return sorted((child for child in path.iterdir() if child.is_dir()), key=lambda child: child.name)


def date_dirs_for_input_root(input_root: Path) -> list[Path]:
    child_dirs = sorted_child_dirs(input_root)
    if not child_dirs:
        return []

    # Support both data/ and data/<date> roots.
    if all(child.name.lower().startswith("box") for child in child_dirs):
        return [input_root]

    return child_dirs


def discover_trial_folders(input_root: Path) -> list[TrialFolder]:
    trial_folders: list[TrialFolder] = []
    for date_dir in date_dirs_for_input_root(input_root):
        for box_dir in sorted_child_dirs(date_dir):
            for trial_dir in sorted_child_dirs(box_dir):
                trial_folders.append(
                    TrialFolder(
                        date=date_dir.name,
                        box=box_dir.name,
                        trial=trial_dir.name,
                        trial_dir=trial_dir,
                    )
                )
    return trial_folders
