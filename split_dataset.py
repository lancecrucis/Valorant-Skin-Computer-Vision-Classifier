"""Create a persistent, source-group-aware train/val/test manifest."""

import argparse
import csv
import hashlib
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from src.classes import WEAPONS

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
NUMBERED_STEM = re.compile(r"^(.*?)[_ .-]?(\d+)$")
SPLITS = ("train", "val", "test")


def source_group(path: Path) -> str:
    """Group numbered sequences together; keep descriptive files independent."""
    match = NUMBERED_STEM.match(path.stem)
    if match:
        prefix = match.group(1).rstrip("_ .-").casefold()
        return f"sequence:{prefix}"
    return f"named:{path.stem.casefold()}"


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_group_splits(
    groups: dict[str, list[Path]], seed: int
) -> dict[str, str]:
    """Assign whole groups, keeping the largest source in training."""
    rng = random.Random(seed)
    items = list(groups.items())
    rng.shuffle(items)
    items.sort(key=lambda item: len(item[1]), reverse=True)
    assignment: dict[str, str] = {}
    counts = Counter({split: 0 for split in SPLITS})
    total = sum(len(paths) for _, paths in items)
    targets = {"train": total * 0.70, "val": total * 0.15, "test": total * 0.15}

    if items:
        largest_name, largest_paths = items.pop(0)
        assignment[largest_name] = "train"
        counts["train"] += len(largest_paths)

    # Give validation and test at least one independent source when possible.
    for split in ("val", "test"):
        if items:
            name, paths = items.pop()
            assignment[name] = split
            counts[split] += len(paths)

    for name, paths in items:
        split = max(SPLITS, key=lambda value: targets[value] - counts[value])
        assignment[name] = split
        counts[split] += len(paths)
    return assignment


def build_manifest(data_dir: Path, output: Path, seed: int) -> None:
    rows: list[dict[str, str]] = []
    digest_locations: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for class_index, class_name in enumerate(WEAPONS):
        class_dir = data_dir / class_name
        paths = sorted(
            (
                path
                for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=lambda path: path.name.casefold(),
        ) if class_dir.exists() else []

        groups: dict[str, list[Path]] = defaultdict(list)
        for path in paths:
            groups[source_group(path)].append(path)
        assignments = choose_group_splits(groups, seed + class_index)

        for group_name, group_paths in groups.items():
            split = assignments[group_name]
            for path in group_paths:
                relative_path = path.relative_to(data_dir).as_posix()
                digest = file_digest(path)
                digest_locations[digest].append((relative_path, split))
                rows.append(
                    {
                        "path": relative_path,
                        "class_name": class_name,
                        "class_index": str(class_index),
                        "source_group": f"{class_name}/{group_name}",
                        "split": split,
                        "sha256": digest,
                    }
                )

    leakage = {
        digest: locations
        for digest, locations in digest_locations.items()
        if len({split for _, split in locations}) > 1
    }
    if leakage:
        examples = next(iter(leakage.values()))
        raise RuntimeError(f"Exact duplicate leakage detected across splits: {examples}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("path", "class_name", "class_index", "source_group", "split", "sha256"),
        )
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["path"].casefold()))

    print(f"Wrote {len(rows)} images to {output}")
    for class_name in WEAPONS:
        class_rows = [row for row in rows if row["class_name"] == class_name]
        counts = Counter(row["split"] for row in class_rows)
        groups = len({row["source_group"] for row in class_rows})
        print(
            f"{class_name:20} groups={groups:3d} "
            f"train={counts['train']:4d} val={counts['val']:3d} test={counts['test']:3d}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output", type=Path, default=Path("splits/split_manifest.csv")
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_manifest(args.data_dir, args.output, args.seed)


if __name__ == "__main__":
    main()
