# Model Card — NeuroAssist AI v2

## Model Details

| Field | Value |
|-------|-------|
| **Architecture** | [FILL: cnn_baseline / resnet50 / efficientnet_b0 / densenet201 / vit_base / swin_tiny / hybrid_cnn_swin] |
| **Version** | [FILL: v1_{arch}_{date}] |
| **Training Date** | [FILL] |
| **Framework** | PyTorch + torchvision + timm |
| **Task** | 4-class Alzheimer's Disease Stage Classification |
| **Input** | 2D brain MRI slices, resized to 224×224, 3-channel |

## Training Data

- **Dataset**: "Best Alzheimer MRI Dataset (99% Accuracy)" by lukechugh on Kaggle
- **Source**: Likely derived from OASIS dataset (2D axial slices)
- **Classes (4)**: Non-Demented, Very Mild Demented, Mild Demented, Moderate Demented
- **Split Strategy**: Train/test split from dataset; patient-level leakage prevention documented

## ⚠️ Known Limitations

1. **Severe Dementia class is NOT covered.** The training data contains only 4 of the 5 clinically recognized stages. The model cannot predict "Severe Dementia." This gap must be disclosed to clinicians.

2. **2D slices, not volumetric 3D MRI.** The model processes single 2D axial slices. Clinical MRI analysis typically involves 3D volume interpretation across multiple planes and sequences.

3. **Limited generalizability.** Trained on a well-worn benchmark dataset (likely OASIS-derived). Real-world clinical accuracy on out-of-distribution scans from different scanners, protocols, and patient populations will likely be lower.

4. **No skull-stripping** in current preprocessing. A proper clinical pipeline would use FSL BET or a pretrained segmentation model for brain extraction.

5. **Class imbalance.** The "Moderate Demented" class has significantly fewer samples than "Non-Demented." Macro-F1 (not just accuracy) is the primary selection metric to account for this.

6. **Not clinically validated.** This model has not undergone prospective clinical validation on a multi-site dataset. It is investigational only.

## Evaluation Metrics

| Metric | Value |
|--------|-------|
| Val Accuracy | [FILL from real evaluation] |
| Val Macro-F1 | [FILL — primary selection metric] |
| Per-class Precision | [FILL from classification_report] |
| Per-class Recall | [FILL from classification_report] |
| Per-class F1 | [FILL from classification_report] |

> All metrics are from a held-out test set. No numbers in the UI may exceed what's reported here.

## Extension Path for "Severe Dementia"

To add a 5th class:
1. Obtain labeled "Severe Dementia" MRI scans from ADNI or OASIS-3 under their data use agreements
2. Re-train with 5-class labels
3. Re-validate on a held-out test set
4. Update this model card with new metrics

This requires institutional data use agreements and cannot be fabricated synthetically.

## Regulatory Status

**Investigational software. Not FDA/CE cleared. Not a substitute for clinical judgment.**
