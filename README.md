# Valorant Skin Classifier

AI-powered Valorant weapon skin bundle classifier with a web interface. Upload any weapon skin screenshot and the model identifies which bundle it belongs to.

## Skin Bundles (27)

reaver, prime, mystbloom, elderflame, glitchpop, nebula, oni, prism, sovereign,
araxys, arcane, spline, smite, ego, g.u.n, singularity, sensation, kuronami,
blackthorn, ayakashi, divergence, prelude to chaos, gaia's vengeance,
neo frontier, cyrax, cyrostasis, forsaken

## Tech Stack

- **Model:** EfficientNet-B0 (transfer learning from ImageNet)
- **Backend:** FastAPI + PyTorch
- **Frontend:** React 18 (CDN), vanilla CSS

## Setup

```bash
uv sync
```

## Run the Website

Start the API server:

```bash
uv run python api.py
```

Then open `frontend/index.html` in your browser (or serve it with any static file server).

## Training

```bash
uv run python split_dataset.py
uv run python train.py
```

The split manifest keeps numbered image sequences together so related frames cannot
appear in both training and evaluation. Training uses validation for model selection
and reports final metrics on the separate test split.

Checkpoints saved to `checkpoints/best_model.pth`.

## CLI Prediction

```bash
uv run python predict.py path/to/image.jpg
```

## Dataset

Place images in `data/<bundle>/` folders:

```
data/
  reaver/
  prime/
  mystbloom/
  elderflame/
  glitchpop/
  nebula/
  oni/
  prism/
  sovereign/
```

Supported formats: `.jpg`, `.jpeg`, `.png`, `.webp`

## Project Structure

```
api.py                  # FastAPI server
train.py                # Training script
predict.py              # CLI inference
frontend/
  index.html            # SPA entry point
  app.js                # React components
  styles.css            # Styling
src/
  dataset.py            # Dataset loader + WEAPONS list
  models/
    classifier.py       # EfficientNet-B0 model
  utils/
    transforms.py       # Image preprocessing
checkpoints/
  best_model.pth        # Trained model weights
  model_meta.json       # Model metadata
```
