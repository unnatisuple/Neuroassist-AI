"""
NeuroAssist AI v2 — Training Script
Unified training loop for all 7 model architectures.
Produces real evaluation reports — no fake/hard-coded metrics.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import json
import hashlib
from pathlib import Path
from datetime import datetime

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

ARCHITECTURES = [
    "cnn_baseline",
    "resnet50",
    "efficientnet_b0",
    "densenet201",
    "vit_base",
    "swin_tiny",
    "hybrid_cnn_swin",
]

CLASS_NAMES = ["Mild_Demented", "Moderate_Demented", "Non_Demented", "Very_Mild_Demented"]
NUM_CLASSES = 4
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transforms():
    """Training and validation transforms."""
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


def build_model(arch_name: str) -> nn.Module:
    """Build model by architecture name."""
    from backend.services.model_service import build_model as _build
    return _build(arch_name, NUM_CLASSES)


def train_model(arch_name: str, train_loader, val_loader, save_dir: str):
    """Train a single model and save checkpoint with real metrics."""
    print(f"\n{'='*60}")
    print(f"Training: {arch_name}")
    print(f"Device: {DEVICE}")
    print(f"{'='*60}")

    model = build_model(arch_name).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_val_f1 = 0.0
    best_epoch = 0

    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                all_preds.extend(predicted.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        # Compute metrics
        from sklearn.metrics import f1_score, classification_report
        val_f1 = f1_score(all_labels, all_preds, average='macro')

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        scheduler.step()

        print(f"  Epoch {epoch+1}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss/train_total:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss/val_total:.4f} | Val Acc: {val_acc:.4f} | "
              f"Val Macro-F1: {val_f1:.4f}")

        # Save best model (by macro-F1, not just accuracy, since classes are imbalanced)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "architecture": arch_name,
                "num_classes": NUM_CLASSES,
                "class_names": CLASS_NAMES,
                "version": f"v1_{arch_name}_{datetime.now().strftime('%Y%m%d')}",
                "val_macro_f1": val_f1,
                "val_accuracy": val_acc,
                "best_epoch": best_epoch,
                "total_epochs": NUM_EPOCHS,
                "training_date": datetime.now().isoformat(),
                "known_limitations": [
                    "Trained on 4-class dataset (Severe Dementia not covered)",
                    "2D MRI slices from Kaggle/OASIS — may not generalize to clinical 3D MRI",
                    "No skull-stripping applied in this version",
                ],
            }

            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, f"{arch_name}_best.pt")
            torch.save(checkpoint, model_path)

    # Final evaluation report
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True)

    # Save evaluation report
    report_dir = os.path.join(str(Path(__file__).parent.parent), "reports")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, f"{arch_name}_evaluation.json")
    with open(report_path, "w") as f:
        json.dump({
            "architecture": arch_name,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_f1,
            "classification_report": report,
            "training_date": datetime.now().isoformat(),
        }, f, indent=2)

    print(f"\n[SUCCESS] {arch_name} - Best Val Macro-F1: {best_val_f1:.4f} (epoch {best_epoch})")
    print(f"   Checkpoint: {save_dir}/{arch_name}_best.pt")
    print(f"   Report: {report_path}")

    return best_val_f1


def main():
    global NUM_EPOCHS, BATCH_SIZE
    import argparse
    parser = argparse.ArgumentParser(description="Train NeuroAssist AI models")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--save-dir", type=str, default="./ml/checkpoints", help="Where to save model checkpoints")
    parser.add_argument("--architectures", nargs="+", default=ARCHITECTURES, help="Which architectures to train")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    NUM_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size

    print("NeuroAssist AI v2 - Model Training Pipeline")
    print(f"   Device: {DEVICE}")
    print(f"   Architectures: {args.architectures}")
    print(f"   Epochs: {NUM_EPOCHS}")
    print(f"   Batch size: {BATCH_SIZE}")

    # Load data
    train_transform, val_transform = get_transforms()

    train_dir = os.path.join(args.data_dir, "train")
    test_dir = os.path.join(args.data_dir, "test")

    if not os.path.exists(train_dir):
        print(f"\n[ERROR] Training data not found at {train_dir}")
        print("   Run download_dataset.py first.")
        sys.exit(1)

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(test_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    print(f"\n   Train: {len(train_dataset)} images")
    print(f"   Val:   {len(val_dataset)} images")
    print(f"   Classes: {train_dataset.classes}")

    # Train all architectures
    results = {}
    for arch in args.architectures:
        try:
            f1 = train_model(arch, train_loader, val_loader, args.save_dir)
            results[arch] = f1
        except Exception as e:
            print(f"\n[ERROR] {arch} training failed: {e}")
            results[arch] = 0.0

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    best_arch = max(results, key=results.get) if results else "none"
    for arch, f1 in sorted(results.items(), key=lambda x: x[1], reverse=True):
        marker = " <- BEST" if arch == best_arch else ""
        print(f"  {arch:25s} | Macro-F1: {f1:.4f}{marker}")

    # Copy best model as the production model
    if best_arch != "none" and results[best_arch] > 0:
        import shutil
        src = os.path.join(args.save_dir, f"{best_arch}_best.pt")
        dst = os.path.join(args.save_dir, "best_model.pt")
        shutil.copy2(src, dst)
        print(f"\n[SUCCESS] Production model: {best_arch} copied to {dst}")
        print(f"   The app will claim metrics from: ml/reports/{best_arch}_evaluation.json")


if __name__ == "__main__":
    main()
