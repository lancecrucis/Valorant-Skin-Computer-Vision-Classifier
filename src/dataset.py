import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.classes import WEAPONS

WEAPON_TO_IDX = {w: i for i, w in enumerate(WEAPONS)}


class ValorantSkinDataset(Dataset):
    """Dataset for Valorant gun skin images organized by weapon category."""

    def __init__(
        self,
        root: str | Path,
        transform: transforms.Compose | None = None,
        manifest: str | Path | None = None,
        split: str | None = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        if manifest is not None:
            if split not in {"train", "val", "test"}:
                raise ValueError("split must be 'train', 'val', or 'test' with a manifest")
            with Path(manifest).open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if row["split"] != split:
                        continue
                    class_name = row["class_name"]
                    if class_name not in WEAPON_TO_IDX:
                        raise ValueError(f"Unknown manifest class: {class_name}")
                    self.samples.append(
                        (self.root / Path(row["path"]), WEAPON_TO_IDX[class_name])
                    )
            return

        for weapon in WEAPONS:
            weapon_dir = self.root / weapon
            if not weapon_dir.exists():
                continue
            for img_path in weapon_dir.iterdir():
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                    self.samples.append((img_path, WEAPON_TO_IDX[weapon]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
