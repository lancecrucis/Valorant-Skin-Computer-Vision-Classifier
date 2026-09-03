from torch import nn
from torchvision import models

from src.classes import WEAPONS

NUM_CLASSES = len(WEAPONS)


def get_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """Return an EfficientNetB0 model fine-tuned for Valorant weapon classification."""
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model
