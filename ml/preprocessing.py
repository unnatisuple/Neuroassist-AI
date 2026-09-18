import os
import random
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

# Repository-relative default dataset directory
ML_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ML_DIR / "data" / "preprocessed"

# Default class order matching directory sorting & backend enum
CLASS_NAMES = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]


class AlzheimerDataset(Dataset):
    """Custom Dataset wrapping image file paths, labels, and transforms."""
    def __init__(
        self,
        filepaths: List[str],
        labels: List[int],
        transform=None,
        targeted_minority_transform=None,
        minority_classes: Optional[List[int]] = None,
        preload: bool = True
    ):
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform
        self.targeted_minority_transform = targeted_minority_transform
        self.minority_classes = minority_classes or []
        self.preload = preload
        self.images = []

        if self.preload:
            for fp in self.filepaths:
                try:
                    with Image.open(fp) as img:
                        self.images.append(img.convert("RGB"))
                except Exception as e:
                    # In case of corrupted file, log error and create fallback black image
                    print(f"[!] Warning: Corrupted image at {fp}: {e}")
                    self.images.append(Image.new("RGB", (176, 208), (0, 0, 0)))

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        label = self.labels[idx]

        if self.preload:
            image = self.images[idx].copy()
        else:
            try:
                with Image.open(self.filepaths[idx]) as img:
                    image = img.convert("RGB")
            except Exception as e:
                print(f"[!] Warning: Error reading {self.filepaths[idx]}: {e}")
                image = Image.new("RGB", (176, 208), (0, 0, 0))

        # Apply targeted minority augmentation if configured and sample belongs to minority
        if self.targeted_minority_transform is not None and label in self.minority_classes:
            image_tensor = self.targeted_minority_transform(image)
        elif self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)

        return image_tensor, label


def get_transforms(img_size: Tuple[int, int] = (208, 176), targeted_aug: bool = False):
    """
    Returns standard train, targeted minority train, and validation/test transforms.
    Images are naturally 176x208 (width x height: 176x208 -> size=(208, 176) in PyTorch (H, W)).
    """
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    # Standard training augmentations
    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])

    # Aggressive targeted augmentation for rare classes (ModerateDemented, MildDemented)
    if targeted_aug:
        minority_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=18),
            transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])
    else:
        minority_transform = None

    eval_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])

    return train_transform, minority_transform, eval_transform


def scan_dataset(data_dir: str | Path) -> Tuple[List[str], List[int], List[str]]:
    """
    Scans the dataset directory and returns filepaths, numeric labels, and class names.
    Supports both flat class folder structure (data_dir/<Class>/*) and
    split structure (data_dir/train/<Class>/*).
    """
    data_dir_str = str(data_dir)
    if not os.path.exists(data_dir_str):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir_str}")

    # Check if this directory directly has class folders
    classes = [d for d in sorted(os.listdir(data_dir_str)) if os.path.isdir(os.path.join(data_dir_str, d)) and d in CLASS_NAMES]

    # If no class folders at top level, check if 'train' directory exists
    scan_roots = [data_dir_str]
    if not classes:
        train_p = os.path.join(data_dir_str, "train")
        if os.path.exists(train_p) and os.path.isdir(train_p):
            sub_classes = [d for d in sorted(os.listdir(train_p)) if os.path.isdir(os.path.join(train_p, d)) and d in CLASS_NAMES]
            if sub_classes:
                classes = sub_classes
                scan_roots = [train_p]
                test_p = os.path.join(data_dir_str, "test")
                if os.path.exists(test_p) and os.path.isdir(test_p):
                    scan_roots.append(test_p)

    if not classes:
        raise ValueError(f"No valid class folders ({CLASS_NAMES}) found in {data_dir_str}")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    filepaths = []
    labels = []

    for root in scan_roots:
        for cls_name in classes:
            cls_dir = os.path.join(root, cls_name)
            if not os.path.exists(cls_dir):
                continue
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    filepaths.append(os.path.join(cls_dir, fname))
                    labels.append(class_to_idx[cls_name])

    return filepaths, labels, classes


def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    """
    Computes balanced class weights:
    weight[c] = total_samples / (num_classes * count[c])
    """
    counts = np.bincount(labels, minlength=num_classes)
    total_samples = len(labels)
    weights = total_samples / (num_classes * counts.astype(np.float32))
    return torch.tensor(weights, dtype=torch.float32)


def build_dataloaders(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    batch_size: int = 32,
    num_workers: int = 0,
    img_size: Tuple[int, int] = (208, 176),
    use_weighted_sampler: bool = True,
    targeted_aug: bool = False,
    train_split: float = 0.70,
    val_split: float = 0.15,
    test_split: float = 0.15,
    preload: bool = True,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor, List[str]]:
    """
    Builds stratified train, validation, and test DataLoaders along with balanced class_weights
    and class_names.

    Returns:
        train_loader, val_loader, test_loader, class_weights, class_names
    """
    filepaths, labels, class_names = scan_dataset(data_dir)
    num_classes = len(class_names)

    # Set seeds for reproducible splitting
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # First split: train vs (val + test)
    val_test_ratio = val_split + test_split
    indices = np.arange(len(labels))
    train_idx, val_test_idx = train_test_split(
        indices,
        test_size=val_test_ratio,
        stratify=labels,
        random_state=seed
    )

    # Second split: val vs test
    val_test_labels = [labels[i] for i in val_test_idx]
    val_relative_ratio = val_split / val_test_ratio
    val_idx_sub, test_idx_sub = train_test_split(
        np.arange(len(val_test_idx)),
        test_size=(1.0 - val_relative_ratio),
        stratify=val_test_labels,
        random_state=seed
    )
    val_idx = val_test_idx[val_idx_sub]
    test_idx = val_test_idx[test_idx_sub]

    # Compute class weights on training set
    train_labels = [labels[i] for i in train_idx]
    class_weights = compute_class_weights(train_labels, num_classes)

    # Transforms
    train_transform, minority_transform, eval_transform = get_transforms(img_size, targeted_aug)

    # Minorities: ModerateDemented (index 1) and MildDemented (index 0)
    minority_classes = [class_names.index("ModerateDemented")]
    if "MildDemented" in class_names:
        minority_classes.append(class_names.index("MildDemented"))

    train_dataset = AlzheimerDataset(
        filepaths=[filepaths[i] for i in train_idx],
        labels=train_labels,
        transform=train_transform,
        targeted_minority_transform=minority_transform,
        minority_classes=minority_classes,
        preload=preload
    )

    val_dataset = AlzheimerDataset(
        filepaths=[filepaths[i] for i in val_idx],
        labels=[labels[i] for i in val_idx],
        transform=eval_transform,
        preload=preload
    )

    test_dataset = AlzheimerDataset(
        filepaths=[filepaths[i] for i in test_idx],
        labels=[labels[i] for i in test_idx],
        transform=eval_transform,
        preload=preload
    )

    # Sampler for training
    if use_weighted_sampler:
        sample_weights = [class_weights[lbl].item() for lbl in train_labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=False
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    return train_loader, val_loader, test_loader, class_weights, class_names


if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_weights, class_names = build_dataloaders()
    print("Class names:", class_names)
    print("Class weights:", class_weights)
    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))
    print("Test batches:", len(test_loader))
    for x, y in train_loader:
        print("Sample batch X shape:", x.shape, "y shape:", y.shape)
        break
