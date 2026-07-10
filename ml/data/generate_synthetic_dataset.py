"""
NeuroAssist AI v2 — Synthetic Dataset Generator
Generates dummy brain scan slices to verify data pipelines and train demo model.
"""

import os
import cv2
import numpy as np
from pathlib import Path

CLASSES = ["Mild_Demented", "Moderate_Demented", "Non_Demented", "Very_Mild_Demented"]
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"


def generate_brain_slice(stage: str, slice_index: int) -> np.ndarray:
    """Create a synthetic 2D axial MRI brain slice (grayscale, 256x256)."""
    # Create dark background
    img = np.zeros((256, 256), dtype=np.uint8)

    # Center and radii for skull / brain
    center = (128, 128)
    axes_skull = (75, 95)
    axes_brain = (70, 90)

    # Draw skull contour
    cv2.ellipse(img, center, axes_skull, 0, 0, 360, 40, -1)
    # Draw brain contour
    cv2.ellipse(img, center, axes_brain, 0, 0, 360, 180, -1)

    # Add ventricular system in the center
    # Mild stages have larger ventricles (atrophy)
    ventricle_size = 8
    if stage == "Mild_Demented":
        ventricle_size = 14
    elif stage == "Moderate_Demented":
        ventricle_size = 20
    elif stage == "Very_Mild_Demented":
        ventricle_size = 11

    # Draw central ventricles
    cv2.circle(img, (115, 128), ventricle_size, 20, -1)
    cv2.circle(img, (141, 128), ventricle_size, 20, -1)

    # Add random brain tissue texture
    noise = np.random.randint(-15, 15, (256, 256), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Re-enforce dark background outside brain
    mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.ellipse(mask, center, axes_brain, 0, 0, 360, 255, -1)
    img = cv2.bitwise_and(img, mask)

    return img


def generate_dataset():
    print("=" * 60)
    print("NeuroAssist AI v2 — Synthetic Dataset Generator")
    print("=" * 60)
    print(f"Target raw directory: {RAW_DIR.resolve()}")

    for cls in CLASSES:
        cls_dir = RAW_DIR / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        print(f"Generating slices for class: {cls}...")

        # Generate 15 scans, with 3 sequential slices each (45 total images per class)
        # Sequence name pattern OAS1_00XX to support OASIS naming prefix hash
        for subject_num in range(1, 16):
            subject_id = f"OAS1_99{subject_num:02d}"
            for slice_num in range(1, 4):
                filename = f"{subject_id}_MR1_mpr-1_{100 + slice_num}.jpg"
                img = generate_brain_slice(cls, slice_num)

                dest_path = cls_dir / filename
                cv2.imwrite(str(dest_path), img)

    print("\n[SUCCESS] Synthetic dataset generated successfully!")
    print("   Ready to run: python ml/data/preprocess.py")


if __name__ == "__main__":
    generate_dataset()
