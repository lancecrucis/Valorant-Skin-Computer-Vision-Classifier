"""Predict weapon category from a Valorant skin image."""

from pathlib import Path

import torch
from PIL import Image

from src.dataset import WEAPONS
from src.models.classifier import get_model
from src.utils.transforms import val_transform

CHECKPOINT_PATH = Path("checkpoints/best_model.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def predict(image_path: str) -> dict[str, float]:
    """Return predicted weapon class and confidence scores."""
    model = get_model(pretrained=False)
    model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    )
    model.to(DEVICE)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    tensor = val_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze()

    scores = {WEAPONS[i]: round(probs[i].item(), 4) for i in range(len(WEAPONS))}
    predicted = WEAPONS[probs.argmax().item()]
    return {"predicted": predicted, "scores": scores}


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: uv run python predict.py <image_path>")
        sys.exit(1)

    result = predict(sys.argv[1])
    print(f"Predicted: {result['predicted']}")
    for weapon, score in sorted(result["scores"].items(), key=lambda x: -x[1]):
        print(f"  {weapon}: {score:.2%}")
