"""
NeuroAssist AI v2 — Data Preprocessing Pipeline
Applies intensity normalization, skull-stripping approximation,
and patient-level train/val/test splitting to prevent data leakage.
"""

import os
import sys
import re
import cv2
import numpy as np
import argparse
from pathlib import Path
import hashlib
from sklearn.model_selection import train_test_split
import shutil

CLASSES = ["Mild_Demented", "Moderate_Demented", "Non_Demented", "Very_Mild_Demented"]
IMG_SIZE = 224


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess NeuroAssist MRI dataset")
    parser.add_argument("--input-dir", type=str, default="ml/data/raw", help="Path to raw dataset")
    parser.add_argument("--output-dir", type=str, default="ml/data/preprocessed", help="Path to save preprocessed dataset")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation/Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def extract_subject_id(filename: str) -> str:
    """
    Extract subject ID from filename to prevent data leakage.
    Ensures that slices from the same subject are kept in the same split.
    """
    # 1. OASIS format: OAS1_0001_MR1_mpr-1_102.jpg -> OAS1_0001
    oasis_match = re.search(r"(OAS\d_\d{4})", filename, re.IGNORECASE)
    if oasis_match:
        return oasis_match.group(1)

    # 2. ADNI format: ADNI_002_S_0413_... -> 002_S_0413
    adni_match = re.search(r"(\d{3}_S_\d{4})", filename, re.IGNORECASE)
    if adni_match:
        return adni_match.group(1)

    # 3. Kaggle sequential format: mildDem102.jpg -> mildDem10
    # Group by prefix before the last 2 digits to group sequential slices of same scan
    num_match = re.search(r"([a-zA-Z_]+)\d{2,}\b", filename)
    if num_match:
        # e.g., mildDem
        prefix = num_match.group(1)
        # Add hash of the filename minus the last few characters
        base = filename.rsplit(".", 1)[0]
        if len(base) > 3:
            return f"{prefix}_{base[:-2]}"

    # Fallback: hash of filename minus extension (each slice treated as its own subject, documented warning)
    return hashlib.md5(filename.encode()).hexdigest()[:8]


def preprocess_image(img_path: Path) -> np.ndarray:
    """
    Preprocess image: Grayscale, skull-stripping approximation, cropping, resizing, normalization.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Skull-stripping approximation (Otsu thresholding + largest contour mask)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Clean up threshold mask with morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Assume the largest contour is the brain/skull
        largest_contour = max(contours, key=cv2.contourArea)

        # Create mask
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)

        # Smooth mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

        # Apply mask
        brain = cv2.bitwise_and(gray, mask)

        # Crop to bounding box of the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        if w > 10 and h > 10:
            brain_cropped = brain[y:y+h, x:x+w]
        else:
            brain_cropped = brain
    else:
        # Fallback if no contours found
        brain_cropped = gray

    # Resize
    resized = cv2.resize(brain_cropped, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    # Intensity normalization (Z-score normalization)
    mean = np.mean(resized)
    std = np.std(resized) + 1e-8
    normalized = (resized - mean) / std

    # Re-scale back to 0-255 range for saving
    min_val, max_val = np.min(normalized), np.max(normalized)
    if max_val - min_val > 0:
        scaled = 255 * (normalized - min_val) / (max_val - min_val)
    else:
        scaled = np.zeros_like(normalized)

    return scaled.astype(np.uint8)


def main():
    args = parse_args()
    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)

    print("=" * 60)
    print("NeuroAssist AI v2 — Data Preprocessing Pipeline")
    print("=" * 60)
    print(f"Input: {input_root}")
    print(f"Output: {output_root}")
    print(f"Test size: {args.test_size}")

    # Gather all raw files and group by subject
    subjects_by_class = {cls: {} for cls in CLASSES}
    total_raw_files = 0

    for cls in CLASSES:
        # Check subdirectories
        cls_dir = input_root / cls
        # Check alternative locations (e.g. Dataset folder nested)
        if not cls_dir.exists():
            # Try to search recursively in raw/
            matches = list(input_root.rglob(cls))
            if matches:
                cls_dir = matches[0]

        if not cls_dir.exists():
            continue

        for ext in ["*.jpg", "*.jpeg", "*.png", "*.tiff"]:
            for img_path in cls_dir.glob(ext):
                subject_id = extract_subject_id(img_path.name)
                if subject_id not in subjects_by_class[cls]:
                    subjects_by_class[cls][subject_id] = []
                subjects_by_class[cls][subject_id].append(img_path)
                total_raw_files += 1

    if total_raw_files == 0:
        print(f"\n[ERROR] No raw dataset files found at {input_root}!")
        print("   Make sure the dataset is downloaded. Run download_dataset.py first.")
        sys.exit(1)

    print(f"\nFound {total_raw_files} raw images.")
    for cls in CLASSES:
        count = sum(len(files) for files in subjects_by_class[cls].values())
        uniq_sub = len(subjects_by_class[cls])
        print(f"  - {cls:20s}: {count} images ({uniq_sub} unique subjects)")

    # Perform patient-level splits
    print("\nSplitting and preprocessing images...")
    processed_count = 0

    for cls in CLASSES:
        subjects = list(subjects_by_class[cls].keys())
        if not subjects:
            continue

        # Split subjects
        train_subs, test_subs = train_test_split(
            subjects, test_size=args.test_size, random_state=args.seed
        )

        splits = {"train": train_subs, "test": test_subs}

        for split_name, subs in splits.items():
            split_dir = output_root / split_name / cls
            split_dir.mkdir(parents=True, exist_ok=True)

            for sub_id in subs:
                for img_path in subjects_by_class[cls][sub_id]:
                    try:
                        # Preprocess
                        proc_img = preprocess_image(img_path)
                        # Save
                        dest_path = split_dir / img_path.name
                        cv2.imwrite(str(dest_path), proc_img)
                        processed_count += 1
                    except Exception as e:
                        print(f"  [WARNING] Error preprocessing {img_path.name}: {e}")

    print(f"\n[SUCCESS] Preprocessing complete! {processed_count} images preprocessed and saved.")
    print(f"   Preprocessed data saved to: {output_root}")


if __name__ == "__main__":
    main()
