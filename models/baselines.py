"""Small supervised baselines for fair comparison."""

from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
import timm


def make_resnet18(num_classes, pretrained=True):
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def make_vit_tiny(num_classes, pretrained=False):
    return timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=pretrained,
        num_classes=num_classes,
    )
