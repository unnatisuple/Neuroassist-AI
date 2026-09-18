import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, Any, List

# Add directory to sys.path for local/repo execution
ML_DIR = Path(__file__).resolve().parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

try:
    from train import train
    from evaluate import run_evaluation
    from preprocessing import DEFAULT_DATA_DIR
except ImportError:
    from ml.train import train
    from ml.evaluate import run_evaluation
    from ml.preprocessing import DEFAULT_DATA_DIR

DEFAULT_SAVE_DIR = ML_DIR / "checkpoints"
DEFAULT_RESULTS_DIR = ML_DIR / "results"


class Config:
    def __init__(self, **kwargs):
        self.model = kwargs.get("model", "cnn")
        self.loss = kwargs.get("loss", "ce")
        self.focal_gamma = kwargs.get("focal_gamma", 2.0)
        self.lr = kwargs.get("lr", None)
        self.weight_decay = kwargs.get("weight_decay", 1e-4)
        self.dropout = kwargs.get("dropout", 0.4)
        self.batch_size = kwargs.get("batch_size", 64)
        self.epochs = kwargs.get("epochs", 12)
        self.patience = kwargs.get("patience", 6)
        self.scheduler = kwargs.get("scheduler", "plateau")
        self.no_weighted_sampler = kwargs.get("no_weighted_sampler", False)
        self.targeted_aug = kwargs.get("targeted_aug", False)
        self.pretrained = kwargs.get("pretrained", True)
        self.data_dir = kwargs.get("data_dir", str(DEFAULT_DATA_DIR))
        self.save_dir = kwargs.get("save_dir", str(DEFAULT_SAVE_DIR))
        self.tag = kwargs.get("tag", "exp")
        self.seed = kwargs.get("seed", 42)
        self.num_workers = kwargs.get("num_workers", 0)
        self.force_cpu = kwargs.get("force_cpu", False)


def run_single_experiment(cfg: Config, use_tta: bool = False, results_dir: Path | str = DEFAULT_RESULTS_DIR) -> Dict[str, Any]:
    print(f"\n{'='*70}")
    print(f" STARTING EXPERIMENT: {cfg.tag} (Model: {cfg.model}, Loss: {cfg.loss}, LR: {cfg.lr}, Aug: {cfg.targeted_aug})")
    print(f"{'='*70}\n")

    start_time = time.time()
    train_res = train(cfg)
    train_time = time.time() - start_time

    # Run evaluation on held-out test set
    eval_res = run_evaluation(
        checkpoint_path=train_res["checkpoint_path"],
        data_dir=cfg.data_dir,
        use_tta=use_tta,
        output_dir=results_dir,
        tag=cfg.tag
    )

    class_names = eval_res.get("class_names", [])
    mod_idx = class_names.index("ModerateDemented") if "ModerateDemented" in class_names else 1
    mod_recall = eval_res["per_class_recall"][mod_idx] if mod_idx < len(eval_res["per_class_recall"]) else 0.0

    combined = {
        "tag": cfg.tag,
        "model": cfg.model,
        "loss": cfg.loss,
        "lr": cfg.lr,
        "dropout": cfg.dropout,
        "batch_size": cfg.batch_size,
        "epochs_trained": len(train_res["history"]),
        "best_epoch": train_res["best_epoch"],
        "train_time_sec": round(train_time, 1),
        "val_macro_f1": round(train_res["best_val_macro_f1"], 4),
        "test_accuracy": round(eval_res["accuracy"], 4),
        "test_macro_f1": round(eval_res["macro_f1"], 4),
        "test_per_class_recall": [round(r, 4) for r in eval_res["per_class_recall"]],
        "test_moderate_demented_recall": round(mod_recall, 4),
        "checkpoint_path": train_res["checkpoint_path"],
        "confusion_matrix_path": eval_res["confusion_matrix_path"],
        "use_tta": use_tta
    }
    return combined


def main():
    parser = argparse.ArgumentParser(description="Run Systematic Alzheimer's MRI Experiments")
    parser.add_argument("--epochs", type=int, default=8, help="Epochs per run for main levers")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience")
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR), help="Path to preprocessed dataset")
    parser.add_argument("--save_dir", type=str, default=str(DEFAULT_SAVE_DIR), help="Path to checkpoint directory")
    parser.add_argument("--results_dir", type=str, default=str(DEFAULT_RESULTS_DIR), help="Path to results directory")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU training")
    parser.add_argument("--lever", type=str, default="all", choices=["all", "a", "b", "c", "d", "e"], help="Run specific lever")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    all_results = []
    data_dir = args.data_dir
    save_dir = args.save_dir
    results_dir = args.results_dir

    # -------------------------------------------------------------
    # LEVER A: Baseline From-Scratch CNN
    # -------------------------------------------------------------
    if args.lever in ["all", "a"]:
        cfg_a = Config(
            model="cnn",
            loss="ce",
            lr=1e-3,
            dropout=0.4,
            batch_size=64,
            epochs=args.epochs,
            patience=args.patience,
            data_dir=data_dir,
            save_dir=save_dir,
            tag="lever_a_cnn_baseline",
            force_cpu=args.force_cpu
        )
        res_a = run_single_experiment(cfg_a, use_tta=False, results_dir=results_dir)
        all_results.append(res_a)

    # -------------------------------------------------------------
    # LEVER B: Backbone Architecture (Pretrained ResNet-18)
    # -------------------------------------------------------------
    if args.lever in ["all", "b"]:
        cfg_b = Config(
            model="resnet",
            loss="ce",
            lr=1e-4,
            dropout=0.4,
            batch_size=64,
            epochs=args.epochs,
            patience=args.patience,
            pretrained=True,
            data_dir=data_dir,
            save_dir=save_dir,
            tag="lever_b_resnet18",
            force_cpu=args.force_cpu
        )
        res_b = run_single_experiment(cfg_b, use_tta=False, results_dir=results_dir)
        all_results.append(res_b)

    # -------------------------------------------------------------
    # LEVER C: Class Imbalance & Loss Function
    # -------------------------------------------------------------
    if args.lever in ["all", "c"]:
        # C1: Focal Loss
        cfg_c_focal = Config(
            model="resnet",
            loss="focal",
            focal_gamma=2.0,
            lr=1e-4,
            dropout=0.4,
            batch_size=64,
            epochs=args.epochs,
            patience=args.patience,
            data_dir=data_dir,
            save_dir=save_dir,
            tag="lever_c_focal_loss",
            force_cpu=args.force_cpu
        )
        res_c1 = run_single_experiment(cfg_c_focal, use_tta=False, results_dir=results_dir)
        all_results.append(res_c1)

        # C2: Targeted Minority Augmentation
        cfg_c_aug = Config(
            model="resnet",
            loss="ce",
            lr=1e-4,
            dropout=0.4,
            batch_size=64,
            epochs=args.epochs,
            patience=args.patience,
            targeted_aug=True,
            data_dir=data_dir,
            save_dir=save_dir,
            tag="lever_c_targeted_aug",
            force_cpu=args.force_cpu
        )
        res_c2 = run_single_experiment(cfg_c_aug, use_tta=False, results_dir=results_dir)
        all_results.append(res_c2)

    # -------------------------------------------------------------
    # LEVER D: Test-Time Augmentation (TTA)
    # -------------------------------------------------------------
    if args.lever in ["all", "d"]:
        resnet_ckpt = os.path.join(save_dir, "best_resnet_lever_b_resnet18.pt")
        if os.path.exists(resnet_ckpt):
            print("\nEvaluating ResNet-18 with Test-Time Augmentation (TTA)...")
            eval_tta = run_evaluation(
                checkpoint_path=resnet_ckpt,
                data_dir=data_dir,
                use_tta=True,
                output_dir=results_dir,
                tag="lever_b_resnet18_with_tta",
                force_cpu=args.force_cpu
            )
            class_names = eval_tta.get("class_names", [])
            mod_idx = class_names.index("ModerateDemented") if "ModerateDemented" in class_names else 1
            mod_recall = eval_tta["per_class_recall"][mod_idx] if mod_idx < len(eval_tta["per_class_recall"]) else 0.0

            res_d = {
                "tag": "lever_b_resnet18_with_tta",
                "model": "resnet",
                "use_tta": True,
                "test_accuracy": round(eval_tta["accuracy"], 4),
                "test_macro_f1": round(eval_tta["macro_f1"], 4),
                "test_per_class_recall": [round(r, 4) for r in eval_tta["per_class_recall"]],
                "test_moderate_demented_recall": round(mod_recall, 4),
                "checkpoint_path": resnet_ckpt,
                "confusion_matrix_path": eval_tta["confusion_matrix_path"]
            }
            all_results.append(res_d)

    # -------------------------------------------------------------
    # LEVER E: Hyperparameter Sweep
    # -------------------------------------------------------------
    if args.lever in ["all", "e"]:
        sweep_configs = [
            {"lr": 5e-5, "dropout": 0.3, "batch_size": 32, "tag": "lever_e_sweep_c1_lr5e-05_do0.3_bs32"},
            {"lr": 5e-5, "dropout": 0.5, "batch_size": 64, "tag": "lever_e_sweep_c2_lr5e-05_do0.5_bs64"},
            {"lr": 1e-4, "dropout": 0.3, "batch_size": 64, "tag": "lever_e_sweep_c3_lr0.0001_do0.3_bs64"},
            {"lr": 1e-4, "dropout": 0.5, "batch_size": 64, "tag": "lever_e_sweep_c4_lr0.0001_do0.5_bs64"},
            {"lr": 2e-4, "dropout": 0.4, "batch_size": 64, "tag": "lever_e_sweep_c5_lr0.0002_do0.4_bs64"},
            {"lr": 1e-4, "dropout": 0.4, "batch_size": 128, "tag": "lever_e_sweep_c6_lr0.0001_do0.4_bs128"},
        ]
        for sc in sweep_configs:
            cfg_sweep = Config(
                model="resnet",
                loss="ce",
                lr=sc["lr"],
                dropout=sc["dropout"],
                batch_size=sc["batch_size"],
                epochs=args.epochs,
                patience=args.patience,
                data_dir=data_dir,
                save_dir=save_dir,
                tag=sc["tag"],
                force_cpu=args.force_cpu
            )
            res_sweep = run_single_experiment(cfg_sweep, use_tta=False, results_dir=results_dir)
            all_results.append(res_sweep)

    # Save summary json
    summary_path = os.path.join(results_dir, "all_experiments.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[*] All experiment results saved to: {summary_path}")


if __name__ == "__main__":
    main()
