"""FastAPI server for Valorant skin classification."""

from pathlib import Path

import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from src.dataset import WEAPONS
from src.models.classifier import get_model
from src.utils.transforms import val_transform

CHECKPOINT_PATH = Path("checkpoints/best_model.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="Valorant Skin Classifier API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
try:
    model = get_model(pretrained=False)
    model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    )
    model.to(DEVICE)
    model.eval()
except Exception as e:
    print(f"Warning: Could not load model: {e}")


@app.get("/health")
def health():
    return {"model_loaded": model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded. Please check checkpoints/best_model.pth."}

    image = Image.open(file.file).convert("RGB")
    tensor = val_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze()

    scores = {WEAPONS[i]: round(probs[i].item(), 4) for i in range(len(WEAPONS))}
    predicted = WEAPONS[probs.argmax().item()]

    return {"predicted": predicted, "scores": scores}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
