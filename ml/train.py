import os
import sys
import argparse
import random
import time
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from sklearn.metrics import f1_score, accuracy_score, recall_score

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

DEFAULT_SAVE_DIR = ML_DIR / "checkpoints"


class FocalLoss(nn.Module):
    """
    Focal Loss with focusing parameter gamma and optional class weighting.
    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
    use_amp: bool = False
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad()

        if use_amp and scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list
) -> Dict[str, Any]:
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    total = len(all_targets)
    avg_loss = running_loss / total if total > 0 else 0.0
    acc = accuracy_score(all_targets, all_preds) if total > 0 else 0.0
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0) if total > 0 else 0.0
    per_class_rec = recall_score(all_targets, all_preds, average=None, zero_division=0) if total > 0 else np.zeros(len(class_names))

    # ModerateDemented index
    mod_idx = class_names.index("ModerateDemented") if "ModerateDemented" in class_names else -1
    mod_recall = per_class_rec[mod_idx] if 0 <= mod_idx < len(per_class_rec) else 0.0

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_recall": per_class_rec,
        "moderate_demented_recall": mod_recall,
        "all_preds": np.array(all_preds),
        "all_targets": np.array(all_targets)
    }


def train(args) -> Dict[str, Any]:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    print(f"[*] Training on Device: {device} | AMP: {use_amp} | Model: {args.model.upper()}")

    # 1. Load Data
    train_loader, val_loader, test_loader, class_weights, class_names = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_weighted_sampler=not args.no_weighted_sampler,
        targeted_aug=args.targeted_aug,
        seed=args.seed
    )

    class_weights = class_weights.to(device)
    print(f"[*] Class Names: {class_names}")
    print(f"[*] Class Weights: {class_weights.cpu().numpy().round(3)}")

    # 2. Build Model
    model = build_model(
        model_type=args.model,
        num_classes=len(class_names),
        dropout=args.dropout,
        pretrained=args.pretrained
    ).to(device)

    # 3. Setup Loss
    if args.loss == "focal":
        criterion = FocalLoss(gamma=args.focal_gamma, weight=class_weights)
        print(f"[*] Loss: FocalLoss (gamma={args.focal_gamma}) with class weights")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"[*] Loss: CrossEntropyLoss with class weights")

    # 4. Optimizer & Scheduler
    lr = args.lr if args.lr is not None else (1e-4 if args.model in ["resnet", "efficientnet"] else 1e-3)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=args.weight_decay)

    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    # 5. Training Loop with Early Stopping on Val Macro-F1
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.save_dir, f"best_{args.model}_{args.tag}.pt")

    best_val_macro_f1 = -1.0
    best_epoch = -1
    patience_counter = 0
    history = []

    print(f"[*] Beginning training ({args.epochs} max epochs, patience={args.patience})...\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=use_amp
        )

        val_metrics = evaluate_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            class_names=class_names
        )

        # Step Scheduler
        if args.scheduler == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_metrics["loss"])

        elapsed = time.time() - t0
        val_macro_f1 = val_metrics["macro_f1"]
        mod_recall = val_metrics["moderate_demented_recall"]
        recalls_str = " | ".join([f"{cls}: {r:.3f}" for cls, r in zip(class_names, val_metrics["per_class_recall"])])

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) - "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']*100:.2f}% "
            f"Macro-F1: {val_macro_f1:.4f} | ModDem Recall: {mod_recall:.4f}"
        )
        print(f"   Recalls: [{recalls_str}]")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_macro_f1": val_macro_f1,
            "moderate_demented_recall": mod_recall,
            "per_class_recall": val_metrics["per_class_recall"].tolist()
        })

        # Save best checkpoint by Val Macro-F1
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "model_type": args.model,
                "val_macro_f1": best_val_macro_f1,
                "val_metrics": val_metrics,
                "class_names": class_names,
                "args": vars(args)
            }, checkpoint_path)
            print(f"   ==> [SAVED BEST CHECKPOINT] (Val Macro-F1: {best_val_macro_f1:.4f}) -> {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[!] Early stopping triggered at epoch {epoch} (no val Macro-F1 improvement for {args.patience} epochs).")
                break

    print(f"\n[*] Training Complete. Best Epoch: {best_epoch} | Best Val Macro-F1: {best_val_macro_f1:.4f}")
    return {
        "checkpoint_path": checkpoint_path,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "history": history,
        "class_names": class_names
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train PyTorch Alzheimer's MRI 4-Class Model")
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "resnet", "efficientnet"], help="Model architecture")
    parser.add_argument("--loss", type=str, default="ce", choices=["ce", "focal"], help="Loss function")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Gamma for Focal Loss")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: 1e-3 for cnn, 1e-4 for transfer)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.4, help="Dropout probability")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Max epochs")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience on val macro-F1")
    parser.add_argument("--scheduler", type=str, default="plateau", choices=["plateau", "cosine"], help="LR scheduler")
    parser.add_argument("--no_weighted_sampler", action="store_true", help="Disable WeightedRandomSampler")
    parser.add_argument("--targeted_aug", action="store_true", help="Enable aggressive targeted minority augmentation")
    parser.add_argument("--pretrained", action="store_true", default=True, help="Use pretrained backbone for transfer learning")
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR), help="Dataset directory")
    parser.add_argument("--save_dir", type=str, default=str(DEFAULT_SAVE_DIR), help="Directory to save checkpoints")
    parser.add_argument("--tag", type=str, default="run", help="Run identifier tag")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader num_workers")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU execution")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
