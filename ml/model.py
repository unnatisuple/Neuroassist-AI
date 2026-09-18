import torch
import torch.nn as nn
from torchvision import models


class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d block."""
    def __init__(self, in_channels: int, out_channels: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AlzheimerCNN(nn.Module):
    """
    From-scratch CNN with 4-5 conv blocks with increasing channel depth (32->64->128->256),
    Global Average Pooling, Dropout, and Linear classifier head.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 4, dropout: float = 0.4):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32, pool=True),    # Output: (B, 32, 104, 88)
            ConvBlock(32, 64, pool=True),             # Output: (B, 64, 52, 44)
            ConvBlock(64, 128, pool=True),            # Output: (B, 128, 26, 22)
            ConvBlock(128, 256, pool=True),           # Output: (B, 256, 13, 11)
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class AlzheimerTransferModel(nn.Module):
    """
    Transfer learning model utilizing pretrained ResNet18 or EfficientNet-B0 backbone.
    Early layers are frozen while deeper representation layers and the classifier head are fine-tuned.
    """
    def __init__(
        self,
        backbone_name: str = "resnet",
        num_classes: int = 4,
        dropout: float = 0.4,
        pretrained: bool = True
    ):
        super().__init__()
        self.backbone_name = backbone_name.lower()

        if self.backbone_name in ["resnet", "resnet18"]:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)

            # Freeze initial layers and early residual stages (conv1, bn1, layer1, layer2, layer3)
            for param in backbone.parameters():
                param.requires_grad = False
            for param in backbone.layer4.parameters():
                param.requires_grad = True

            in_features = backbone.fc.in_features
            backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes)
            )
            self.model = backbone

        elif self.backbone_name in ["efficientnet", "efficientnet_b0"]:
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_b0(weights=weights)

            # Freeze early features stages (0 through 5)
            for param in backbone.parameters():
                param.requires_grad = False
            for stage in backbone.features[6:]:
                for param in stage.parameters():
                    param.requires_grad = True

            in_features = backbone.classifier[1].in_features
            backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes)
            )
            self.model = backbone
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}. Choose 'resnet' or 'efficientnet'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_model(
    model_type: str = "cnn",
    num_classes: int = 4,
    dropout: float = 0.4,
    pretrained: bool = True
) -> nn.Module:
    """
    Factory function to construct models:
    --model cnn | resnet | efficientnet
    """
    model_type = model_type.lower()
    if model_type == "cnn":
        return AlzheimerCNN(in_channels=3, num_classes=num_classes, dropout=dropout)
    elif model_type in ["resnet", "resnet18"]:
        return AlzheimerTransferModel(
            backbone_name="resnet18",
            num_classes=num_classes,
            dropout=dropout,
            pretrained=pretrained
        )
    elif model_type in ["efficientnet", "efficientnet_b0"]:
        return AlzheimerTransferModel(
            backbone_name="efficientnet_b0",
            num_classes=num_classes,
            dropout=dropout,
            pretrained=pretrained
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Expected 'cnn', 'resnet', or 'efficientnet'.")


if __name__ == "__main__":
    dummy_input = torch.randn(2, 3, 208, 176)
    for mtype in ["cnn", "resnet", "efficientnet"]:
        model = build_model(mtype, num_classes=4, dropout=0.4, pretrained=False)
        out = model(dummy_input)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[{mtype.upper()}] Output shape: {out.shape}, Total params: {total_params:,}, Trainable: {trainable_params:,}")
