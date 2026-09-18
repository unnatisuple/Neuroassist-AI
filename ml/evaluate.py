import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)

# Add directory to sys.path for local/repo execution
ML_DIR = Path(__file__).resolve().parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

try:
    from preprocessing import build_dataloaders, DEFAULT_DATA_DIR, CLASS_NAMES
    from model import build_model
except ImportError:
    from ml.preprocessing import build_dataloaders, DEFAULT_DATA_DIR, CLASS_NAMES
    from ml.model import build_model

DEFAULT_CHECKPOINT = ML_DIR / "checkpoints" / "best_model.pt"
DEFAULT_OUTPUT_DIR = ML_DIR / "results"


def predict_batch_standard(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    outputs = model(images)
    probs = torch.softmax(outputs, dim=1)
    return probs


def predict_batch_tta(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """
    Test-Time Augmentation (TTA):
    Averages predictions across:
    1. Original view
    2. Horizontally flipped view
    3. +5 degree rotated view
    4. -5 degree rotated view
    """
    views = [
        images,
        torch.flip(images, dims=[-1]),
        TF.rotate(images, angle=5.0),
        TF.rotate(images, angle=-5.0),
    ]

    all_probs = []
    for view in views:
        out = model(view)
        probs = torch.softmax(out, dim=1)
        all_probs.append(probs)

    avg_probs = torch.stack(all_probs, dim=0).mean(dim=0)
    return avg_probs


@torch.no_grad()
def evaluate_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    use_tta: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    for images, targets in test_loader:
        images = images.to(device)
        if use_tta:
            probs = predict_batch_tta(model, images)
        else:
            probs = predict_batch_standard(model, images)

        preds = torch.argmax(probs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_targets), np.array(all_probs)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str,
    title: str = "Confusion Matrix"
):
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label"
    )

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            row_sum = cm[i].sum()
            pct = (val / row_sum * 100) if row_sum > 0 else 0
            ax.text(
                j,
                i,
                f"{val}\n({pct:.1f}%)",
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontsize=9
            )

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[*] Confusion matrix plot saved to: {save_path}")


def run_evaluation(
    checkpoint_path: str | Path,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    batch_size: int = 32,
    use_tta: bool = False,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    tag: str = "eval",
    force_cpu: bool = False
) -> Dict[str, Any]:
    checkpoint_path = str(checkpoint_path)
    data_dir = str(data_dir)
    output_dir = str(output_dir)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")
    print(f"\n{'='*70}")
    print(f" RUNNING EVALUATION ON TEST SET (Device: {device} | TTA: {use_tta})")
    print(f" Checkpoint: {checkpoint_path}")
    print(f" Data Dir:   {data_dir}")
    print(f"{'='*70}\n")

    # 1. Load Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_type = checkpoint.get("model_type") or checkpoint.get("architecture") or checkpoint.get("args", {}).get("model", "resnet")
    class_names = checkpoint.get("class_names", CLASS_NAMES)
    num_classes = len(class_names)

    # 2. Build Model & Load Weights
    model = build_model(
        model_type=model_type,
        num_classes=num_classes,
        dropout=checkpoint.get("args", {}).get("dropout", 0.4),
        pretrained=False
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[*] Successfully loaded {model_type.upper()} model weights.")

    # 3. Data Loader
    _, _, test_loader, _, _ = build_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=0,
        seed=checkpoint.get("args", {}).get("seed", 42)
    )

    # 4. Predict
    all_preds, all_targets, all_probs = evaluate_test_set(
        model=model,
        test_loader=test_loader,
        device=device,
        use_tta=use_tta
    )

    # 5. Metrics
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    macro_precision = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    macro_recall = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    per_class_rec = recall_score(all_targets, all_preds, average=None, zero_division=0)
    per_class_prec = precision_score(all_targets, all_preds, average=None, zero_division=0)
    per_class_f1 = f1_score(all_targets, all_preds, average=None, zero_division=0)

    report_str = classification_report(all_targets, all_preds, target_names=class_names, digits=4, zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)

    print("\n" + "="*50)
    print(" HELD-OUT TEST SET EVALUATION RESULTS")
    print("="*50)
    print(f"Test Accuracy:       {acc*100:.2f}%")
    print(f"Test Macro-F1:       {macro_f1:.4f}")
    print(f"Test Macro-Precision:{macro_precision:.4f}")
    print(f"Test Macro-Recall:   {macro_recall:.4f}")
    print("\nDetailed Per-Class Classification Report:")
    print(report_str)

    print("Confusion Matrix:")
    print(cm)

    # 6. Save Confusion Matrix Plot
    tta_suffix = "_tta" if use_tta else ""
    cm_path = os.path.join(output_dir, f"confusion_matrix_{model_type}_{tag}{tta_suffix}.png")
    plot_confusion_matrix(
        cm=cm,
        class_names=class_names,
        save_path=cm_path,
        title=f"Test Confusion Matrix ({model_type.upper()}{tta_suffix.upper()}) Acc: {acc*100:.1f}%"
    )

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "per_class_recall": per_class_rec.tolist(),
        "per_class_precision": per_class_prec.tolist(),
        "per_class_f1": per_class_f1.tolist(),
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "confusion_matrix_path": cm_path,
        "all_preds": all_preds,
        "all_targets": all_targets
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PyTorch Alzheimer's MRI Model")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT), help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR), help="Dataset directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--use_tta", action="store_true", help="Enable Test-Time Augmentation")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory for plots")
    parser.add_argument("--tag", type=str, default="eval", help="Tag for output artifacts")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU execution")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        use_tta=args.use_tta,
        output_dir=args.output_dir,
        tag=args.tag,
        force_cpu=args.force_cpu
    )
