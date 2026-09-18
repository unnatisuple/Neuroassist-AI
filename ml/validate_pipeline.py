"""
Comprehensive validation test for NeuroAssist AI v2 ML Pipeline.
Validates:
1. Dataset scan and exact class counts (6,400 total)
2. Preprocessing & Dataset DataLoader pipeline
3. Model architecture forward passes (CNN & ResNet-18)
4. Checkpoint integrity (best_model.pt)
5. Healthy template loading (healthy_template.npy)
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
from collections import Counter

# Ensure repository root and ml directory are in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "ml"))

from ml.preprocessing import scan_dataset, get_transforms, AlzheimerDataset, CLASS_NAMES
from ml.model import AlzheimerCNN, AlzheimerTransferModel, build_model

def test_dataset_counts():
    print("=== Test 1: Dataset Scan & Class Counts ===")
    data_dir = REPO_ROOT / "ml" / "data" / "preprocessed"
    filepaths, labels, classes = scan_dataset(data_dir)
    total = len(filepaths)
    print(f"Total scans found: {total}")
    print(f"Classes: {classes}")
    
    label_counts = Counter(labels)
    class_counts = {classes[idx]: count for idx, count in label_counts.items()}
    print(f"Class distribution: {class_counts}")
    
    expected_counts = {
        "MildDemented": 896,
        "ModerateDemented": 64,
        "NonDemented": 3200,
        "VeryMildDemented": 2240
    }
    assert total == 6400, f"Expected 6,400 scans, got {total}"
    for cls, count in expected_counts.items():
        assert class_counts.get(cls) == count, f"Mismatch for class {cls}: expected {count}, got {class_counts.get(cls)}"
    print("PASSED: Dataset scan and exact class counts match.\n")
    return filepaths, labels

def test_dataloader(filepaths, labels):
    print("=== Test 2: Preprocessing Transforms & DataLoader ===")
    train_transform, _, eval_transform = get_transforms(img_size=(208, 176))
    dataset = AlzheimerDataset(filepaths[:16], labels[:16], transform=eval_transform, preload=False)
    assert len(dataset) == 16
    img, label = dataset[0]
    print(f"Sample tensor shape: {img.shape}, label: {label} ({CLASS_NAMES[label]})")
    assert img.shape == (3, 208, 176), f"Expected shape (3, 208, 176), got {img.shape}"
    assert isinstance(label, (int, torch.Tensor))
    print("PASSED: Image transformed correctly to (3, 208, 176).\n")

def test_models():
    print("=== Test 3: Model Forward Passes ===")
    dummy = torch.randn(2, 3, 208, 176)
    
    # 1. AlzheimerCNN
    cnn = AlzheimerCNN(in_channels=3, num_classes=4)
    cnn.eval()
    out_cnn = cnn(dummy)
    print(f"AlzheimerCNN output shape: {out_cnn.shape}")
    assert out_cnn.shape == (2, 4), f"Expected (2, 4), got {out_cnn.shape}"
    
    # 2. AlzheimerTransferModel (ResNet-18)
    resnet = AlzheimerTransferModel(backbone_name="resnet18", num_classes=4, pretrained=False)
    resnet.eval()
    out_resnet = resnet(dummy)
    print(f"AlzheimerTransferModel (resnet18) output shape: {out_resnet.shape}")
    assert out_resnet.shape == (2, 4), f"Expected (2, 4), got {out_resnet.shape}"
    print("PASSED: Model forward passes match expected output (B, 4).\n")

def test_checkpoint():
    print("=== Test 4: Checkpoint Loading & Inference ===")
    ckpt_path = REPO_ROOT / "ml" / "checkpoints" / "best_model.pt"
    assert ckpt_path.exists(), f"Checkpoint {ckpt_path} not found"
    
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"Loaded checkpoint keys: {list(ckpt.keys())}")
    
    # Model architecture for best_resnet_lever_b is resnet18
    model = build_model("resnet18", num_classes=4, pretrained=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    
    dummy = torch.randn(1, 3, 208, 176)
    with torch.no_grad():
        out = model(dummy)
        probs = torch.softmax(out, dim=1).squeeze().tolist()
    print(f"Model prediction probabilities on dummy input: {probs}")
    print("PASSED: Checkpoint loads and infers successfully.\n")

def test_healthy_template():
    print("=== Test 5: Healthy Brain Template ===")
    template_path = REPO_ROOT / "ml" / "reports" / "healthy_template.npy"
    assert template_path.exists(), f"Template not found at {template_path}"
    template = np.load(template_path)
    print(f"Template shape: {template.shape}, dtype: {template.dtype}, min: {template.min():.4f}, max: {template.max():.4f}")
    assert template.shape == (224, 224), f"Expected shape (224, 224), got {template.shape}"
    print("PASSED: Healthy brain template verified.\n")

if __name__ == "__main__":
    print("Starting ML Pipeline Validation...\n")
    fps, lbls = test_dataset_counts()
    test_dataloader(fps, lbls)
    test_models()
    test_checkpoint()
    test_healthy_template()
    print("==================================================")
    print("ALL 5 TESTS PASSED SUCCESSFULLY! ML Pipeline is fully verified.")
    print("==================================================")
