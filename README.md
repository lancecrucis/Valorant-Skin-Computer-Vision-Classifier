# Valorant Skin Classifier

Image classifier for Valorant gun skins using transfer learning with ResNet18.

## Categories

- Phantom
- Vandal
- Operator
- Classic
- Ghost
- Sheriff

## Setup

```bash
uv sync
```

## Dataset

Place images in `data/<weapon>/` folders:

```
data/
  phantom/     # Phantom skin images
  vandal/      # Vandal skin images
  operator/    # Operator skin images
  classic/     # Classic skin images
  ghost/       # Ghost skin images
  sheriff/     # Sheriff skin images
```

Supported formats: `.jpg`, `.jpeg`, `.png`, `.webp`

## Training

```bash
uv run python train.py
```

Checkpoints saved to `checkpoints/best_model.pth`.

## Prediction

```bash
uv run python predict.py path/to/image.jpg
```

## Project Structure

```
src/
  dataset.py            # Dataset loader
  models/
    classifier.py       # ResNet18 model
  utils/
    transforms.py       # Image transforms
train.py                # Training script
predict.py              # Inference script
```
