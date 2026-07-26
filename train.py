"""Train a Valorant gun skin classifier using EfficientNetB0 transfer learning."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, random_split

from src.dataset import WEAPONS, ValorantSkinDataset
from src.models.classifier import get_model
from src.utils.transforms import train_transform, val_transform

DATA_DIR = Path("data")
CHECKPOINT_DIR = Path("checkpoints")
OUTPUT_DIR = Path("outputs")
BATCH_SIZE = 32
EPOCHS = 25
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds: list[int] = []
    all_labels: list[int] = []
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    return running_loss / len(loader), correct / total, all_preds, all_labels


def plot_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    output_path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(WEAPONS)),
        yticks=np.arange(len(WEAPONS)),
        xticklabels=WEAPONS,
        yticklabels=WEAPONS,
        ylabel="True",
        xlabel="Predicted",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_training_curves(
    history: dict[str, list[float]],
    output_path: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"], label="Val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="Train")
    ax2.plot(epochs, history["val_acc"], label="Val")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def train() -> None:
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    full_dataset = ValorantSkinDataset(DATA_DIR, transform=train_transform)
    if len(full_dataset) == 0:
        print("No images found in data/. Add images to data/<weapon>/ folders first.")
        return

    print(f"Dataset: {len(full_dataset)} images across {len(WEAPONS)} classes")

    val_size = max(1, int(0.2 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    val_set.dataset = ValorantSkinDataset(DATA_DIR, transform=val_transform)

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    model = get_model(pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
    best_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer
        )
        val_loss, val_acc, _val_preds, _val_labels = evaluate(
            model, val_loader, criterion
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best_model.pth")
            meta = {
                "best_epoch": epoch,
                "best_val_acc": round(best_acc, 4),
                "num_classes": len(WEAPONS),
                "weapons": WEAPONS,
                "image_size": 224,
                "model": "efficientnet_b0",
            }
            (CHECKPOINT_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2))
            print(f"  -> Saved best model (val_acc={best_acc:.3f})")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            print(
                f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)"
            )
            break

    # Reload best model for final evaluation
    model.load_state_dict(
        torch.load(
            CHECKPOINT_DIR / "best_model.pth", map_location=DEVICE, weights_only=True
        )
    )
    _, _final_acc, final_preds, final_labels = evaluate(model, val_loader, criterion)

    plot_training_curves(history, OUTPUT_DIR / "training_curves.png")
    plot_confusion_matrix(
        final_labels, final_preds, OUTPUT_DIR / "confusion_matrix.png"
    )

    report = classification_report(final_labels, final_preds, target_names=WEAPONS)
    (OUTPUT_DIR / "classification_report.txt").write_text(report)
    print(f"\n{report}")

    print(f"Training complete. Best val accuracy: {best_acc:.3f}")
    print(f"Outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    train()
