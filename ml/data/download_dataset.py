"""
NeuroAssist AI v2 — Dataset Download Script
Downloads the Kaggle Alzheimer's MRI dataset using the Kaggle API.
Requires the user's own kaggle.json credentials.
"""

import os
import sys
import zipfile
import shutil
from pathlib import Path


DATASET_SLUG = "lukechugh/best-alzheimer-mri-dataset-99-accuracy"
OUTPUT_DIR = Path(__file__).parent / "raw"
EXPECTED_CLASSES = ["Mild_Demented", "Moderate_Demented", "Non_Demented", "Very_Mild_Demented"]


def download_dataset():
    """Download and extract the Kaggle dataset."""
    print("=" * 60)
    print("NeuroAssist AI v2 — Dataset Download")
    print("=" * 60)

    # Check Kaggle credentials
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("\n[ERROR] Kaggle credentials not found!")
        print(f"   Expected at: {kaggle_json}")
        print("\n   To set up Kaggle API credentials:")
        print("   1. Go to https://www.kaggle.com/account")
        print("   2. Click 'Create New Token' to download kaggle.json")
        print(f"   3. Place it at: {kaggle_json}")
        print("   4. Run this script again")
        sys.exit(1)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("\n[ERROR] Kaggle Python package not installed.")
        print("   Run: pip install kaggle")
        sys.exit(1)

    # Download
    print(f"\n[INFO] Downloading dataset: {DATASET_SLUG}")
    print(f"   Output directory: {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(DATASET_SLUG, path=str(OUTPUT_DIR), unzip=True)

    print("[SUCCESS] Dataset downloaded and extracted successfully!")

    # Verify structure
    print("\n[INFO] Verifying dataset structure...")
    found_classes = []
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for d in dirs:
            if d in EXPECTED_CLASSES:
                found_classes.append(d)
                count = len(list((Path(root) / d).glob("*")))
                print(f"   - {d}: {count} images")

    if len(found_classes) < 4:
        print(f"\n[WARNING] Expected 4 classes, found {len(found_classes)}")
        print(f"   Expected: {EXPECTED_CLASSES}")
        print(f"   Found: {found_classes}")
    else:
        print(f"\n[SUCCESS] All 4 classes found. Dataset ready for preprocessing.")

    print("\n[INFO] IMPORTANT NOTES:")
    print("   - This dataset covers 4 classes. 'Severe Dementia' is NOT included.")
    print("   - The '99% accuracy' in the dataset name is misleading - expect")
    print("     lower real-world accuracy on out-of-distribution scans.")
    print("   - Images are 2D slices, likely from the OASIS dataset.")
    print("   - Run preprocess.py next to prepare training data.")


if __name__ == "__main__":
    download_dataset()
