"""FastAPI server for Valorant skin classification."""

import time
from collections import defaultdict
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

from src.dataset import WEAPONS
from src.models.classifier import get_model
from src.utils.transforms import val_transform

CHECKPOINT_PATH = Path("checkpoints/best_model.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_PIXELS = 4096 * 4096

RATE_LIMIT_WINDOW = 60
RATE_LIMIT_MAX = 30

app = FastAPI(title="Valorant Skin Classifier API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5500",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5500",
        "http://127.0.0.1:8080",
        "null",
    ],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

_rate_limits: dict[str, list[float]] = defaultdict(list)


def _check_rate_limit(client_ip: str) -> None:
    now = time.time()
    window = _rate_limits[client_ip]
    _rate_limits[client_ip] = [t for t in window if now - t < RATE_LIMIT_WINDOW]
    if len(_rate_limits[client_ip]) >= RATE_LIMIT_MAX:
        raise HTTPException(status_code=429, detail="Too many requests. Try again later.")
    _rate_limits[client_ip].append(now)


model = None
model_error = None
try:
    model = get_model(pretrained=False)
    model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    )
    model.to(DEVICE)
    model.eval()
except Exception as e:
    model_error = str(e)
    print(f"Warning: Could not load model: {e}")


@app.get("/health")
def health():
    return {"model_loaded": model is not None, "error": model_error}


@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    _check_rate_limit(request.client.host if request.client else "unknown")

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check server logs.")

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Allowed: {', '.join(ALLOWED_TYPES)}",
        )

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(contents) // (1024*1024)}MB). Max size: {MAX_FILE_SIZE // (1024*1024)}MB.",
        )

    try:
        import io

        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image. File may be corrupted.")

    if image.size[0] * image.size[1] > MAX_IMAGE_PIXELS:
        raise HTTPException(
            status_code=400,
            detail=f"Image dimensions too large ({image.size[0]}x{image.size[1]}). Max: {int(MAX_IMAGE_PIXELS**0.5)}px per side.",
        )

    try:
        tensor = val_transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze()

        scores = {WEAPONS[i]: round(probs[i].item(), 4) for i in range(len(WEAPONS))}
        predicted = WEAPONS[probs.argmax().item()]

        return {"predicted": predicted, "scores": scores}
    except Exception:
        raise HTTPException(status_code=500, detail="Prediction failed. Check server logs.")


app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
