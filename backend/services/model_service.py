"""
NeuroAssist AI v2 — Model Service
Build model architectures from checkpoint metadata.
"""

import torch
import torch.nn as nn


def build_model(architecture: str, num_classes: int = 4) -> nn.Module:
    """
    Reconstruct a model architecture from its name.
    Used when loading checkpoints that contain only state_dict + metadata.
    """
    arch = architecture.lower()

    if arch == "resnet50":
        import torchvision.models as models
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    elif arch == "efficientnet_b0":
        import timm
        model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)
        return model

    elif arch == "densenet201":
        import torchvision.models as models
        model = models.densenet201(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model

    elif arch == "vit_base":
        import timm
        model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes)
        return model

    elif arch == "swin_tiny":
        import timm
        model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=num_classes)
        return model

    elif arch == "cnn_baseline":
        return CNNBaseline(num_classes)

    elif arch == "hybrid_cnn_swin":
        return HybridCNNSwin(num_classes)

    else:
        raise ValueError(f"Unknown architecture: {architecture}")


class CNNBaseline(nn.Module):
    """Custom 5-layer CNN baseline for Alzheimer's classification."""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class HybridCNNSwin(nn.Module):
    """
    Hybrid CNN + Swin Transformer architecture.
    CNN feature extractor feeds into a Swin Transformer head.
    """

    def __init__(self, num_classes: int = 4):
        super().__init__()
        # CNN feature extractor (lightweight)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 3, 1),  # Project back to 3 channels for Swin input
        )

        # Swin Transformer head
        import timm
        self.swin = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=False,
            num_classes=num_classes,
        )

    def forward(self, x):
        # CNN extracts features, then Swin processes them
        features = self.cnn(x)
        # Resize to 224x224 if needed
        if features.shape[-2:] != (224, 224):
            features = nn.functional.interpolate(features, size=(224, 224), mode="bilinear", align_corners=False)
        return self.swin(features)
