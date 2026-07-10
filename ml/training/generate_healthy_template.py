import os
import sys
import numpy as np
import cv2
from PIL import Image

def generate():
    print("=" * 60)
    print("NeuroAssist AI v2 — Healthy Brain Template Generator")
    print("=" * 60)
    
    non_demented_dir = "ml/data/preprocessed/train/Non_Demented"
    if not os.path.exists(non_demented_dir):
        print(f"[ERROR] Non-Demented preprocessed directory not found at: {non_demented_dir}")
        print("Please run preprocessing first: python ml/data/preprocess.py")
        sys.exit(1)
        
    images = []
    for f in os.listdir(non_demented_dir):
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".tiff")):
            img_path = os.path.join(non_demented_dir, f)
            try:
                img = Image.open(img_path).convert("L").resize((224, 224))
                img_array = np.array(img).astype(np.float32) / 255.0
                images.append(img_array)
            except Exception as e:
                print(f"  [WARNING] Could not read {f}: {e}")
                
    if len(images) == 0:
        print("[ERROR] No images found in Non-Demented folder.")
        sys.exit(1)
        
    template = np.mean(images, axis=0)
    
    # Save template to ml/reports/
    os.makedirs("ml/reports", exist_ok=True)
    template_path = "ml/reports/healthy_template.npy"
    np.save(template_path, template)
    print(f"\n[SUCCESS] Healthy template generated successfully from {len(images)} Non-Demented images!")
    print(f"   Saved to: {template_path}")

if __name__ == "__main__":
    generate()
