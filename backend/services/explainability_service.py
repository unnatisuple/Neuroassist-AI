"""
NeuroAssist AI v2 — Explainability Service
Generates XAI visualizations from REAL model activations on REAL uploaded scans.
Never returns stock/sample heatmap images.
Supports: Grad-CAM, Grad-CAM++, Integrated Gradients (Captum), and 2D Image-Space Brain Region Analysis.
"""

import numpy as np
import base64
import io
from PIL import Image
from torchvision import transforms
from typing import Dict, Any, Optional, List
import cv2

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _img_to_base64(img_array: np.ndarray) -> str:
    """Convert numpy array to base64-encoded PNG."""
    if img_array.dtype != np.uint8:
        img_array = (np.clip(img_array, 0.0, 1.0) * 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _find_target_layer(model):
    """Find the last convolutional layer for Grad-CAM targeting."""
    import torch.nn as nn

    target_layer = None
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d,)):
            target_layer = module

    if target_layer is None:
        # For transformers, try to find the norm layer before head
        for name, module in model.named_modules():
            if "norm" in name.lower() and isinstance(module, nn.LayerNorm):
                target_layer = module

    if target_layer is None:
        raise ValueError("Could not find a suitable target layer for Grad-CAM.")

    return target_layer


def _generate_gradcam(
    model,
    input_tensor,
    original_img,
    method: str,
    target_class_idx: Optional[int] = None,
) -> dict:
    """
    Generate Grad-CAM / Grad-CAM++ / HiResCAM via pytorch-grad-cam.
    Computed directly from CNN activations and gradients for the target class.
    """
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, HiResCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    import torch

    # Find the target layer (last conv layer)
    target_layer = _find_target_layer(model)

    cam_class = {
        "gradcam": GradCAM,
        "gradcam++": GradCAMPlusPlus,
        "hirescam": HiResCAM,
    }[method]

    cam = cam_class(model=model, target_layers=[target_layer])

    # Determine target class index
    if target_class_idx is None:
        with torch.no_grad():
            output = model(input_tensor.detach())
            target_class_idx = int(torch.argmax(output, dim=1).item())

    targets = [ClassifierOutputTarget(target_class_idx)]
    grayscale_cam = cam(input_tensor=input_tensor.detach(), targets=targets)[0]

    # Create overlay on original image
    img_array = np.array(original_img).astype(np.float32) / 255.0
    overlay = show_cam_on_image(img_array, grayscale_cam, use_rgb=True)

    # Heatmap only (as JET color map)
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    return {
        "heatmap_base64": _img_to_base64(heatmap_colored),
        "overlay_base64": _img_to_base64(overlay),
        "raw_cam": grayscale_cam,
    }


def _generate_integrated_gradients(
    model,
    input_tensor,
    original_img,
    target_class_idx: Optional[int] = None,
) -> dict:
    """
    Generate Integrated Gradients attribution via Captum.
    Calculates path-integral attribution from a black zero baseline to the input image.
    
    Baseline Documentation:
    baseline = torch.zeros_like(input_tensor):
    A zero-tensor baseline representing total absence of visual MRI signal in normalized pixel space.
    """
    from captum.attr import IntegratedGradients
    import torch

    ig = IntegratedGradients(model)
    baseline = torch.zeros_like(input_tensor)

    if target_class_idx is None:
        with torch.no_grad():
            output = model(input_tensor.detach())
            target_class_idx = int(torch.argmax(output, dim=1).item())

    attributions = ig.attribute(
        input_tensor,
        baselines=baseline,
        target=target_class_idx,
        n_steps=25,
    )

    # Convert attributions to spatial attribution heatmap
    attr_np = attributions.squeeze().detach().cpu().numpy()
    attr_magnitude = np.abs(attr_np).sum(axis=0)  # Sum across channels
    attr_normalized = attr_magnitude / (attr_magnitude.max() + 1e-8)

    heatmap = cv2.applyColorMap(np.uint8(255 * attr_normalized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # 50/50 alpha blend overlay on original image
    img_array = np.array(original_img).astype(np.float32) / 255.0
    overlay_array = img_array * 0.5 + (heatmap.astype(np.float32) / 255.0) * 0.5

    return {
        "heatmap_base64": _img_to_base64(heatmap),
        "overlay_base64": _img_to_base64(overlay_array),
        "raw_attribution": attr_normalized,
    }


def _generate_guided_backprop(
    model,
    input_tensor,
    original_img,
    target_class_idx: Optional[int] = None,
) -> dict:
    """Generate Guided Backpropagation via Captum."""
    from captum.attr import GuidedBackprop
    import torch

    gbp = GuidedBackprop(model)

    if target_class_idx is None:
        with torch.no_grad():
            output = model(input_tensor.detach())
            target_class_idx = int(torch.argmax(output, dim=1).item())

    attributions = gbp.attribute(input_tensor, target=target_class_idx)

    attr_np = attributions.squeeze().detach().cpu().numpy()
    attr_magnitude = np.abs(attr_np).sum(axis=0)
    attr_normalized = attr_magnitude / (attr_magnitude.max() + 1e-8)

    heatmap = cv2.applyColorMap(np.uint8(255 * attr_normalized), cv2.COLORMAP_INFERNO)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    img_array = np.array(original_img).astype(np.float32) / 255.0
    overlay_array = img_array * 0.5 + (heatmap.astype(np.float32) / 255.0) * 0.5

    return {
        "heatmap_base64": _img_to_base64(heatmap),
        "overlay_base64": _img_to_base64(overlay_array),
        "raw_attribution": attr_normalized,
    }


def analyze_brain_regions(attribution_map: np.ndarray) -> dict:
    """
    Calculate 2D image-space anatomical brain region attribution from model explanation map.
    
    IMPORTANT CLINICAL ACCURACY REQUIREMENT:
    Since 2D axial MRI slices lack 3D stereotaxic coordinates / non-rigid atlas registration,
    this performs canonical axial 2D spatial prior mapping and is transparently labeled as:
    'Estimated anatomical region based on 2D axial image-space localization; not a clinical 3D anatomical segmentation.'
    
    Supported Regions (correct anatomical spelling):
    - Frontal Lobe
    - Parietal Lobe
    - Temporal Lobe
    - Occipital Lobe
    - Entorhinal Cortex
    - Hippocampus
    - Cingulate Cortex
    - Precuneus
    """
    h, w = attribution_map.shape[:2]
    y, x = np.ogrid[:h, :w]

    def gaussian_mask(cy: float, cx: float, sy: float, sx: float) -> np.ndarray:
        return np.exp(-(((y - cy)**2 / (2 * sy**2)) + ((x - cx)**2 / (2 * sx**2))))

    # Canonical 2D axial stereotaxic spatial priors (scaled to h=224, w=224)
    # Axial orientation: anterior (top, small y), posterior (bottom, large y)
    regions_config = {
        "Frontal Lobe": {
            "mask": gaussian_mask(65, 112, 28, 45),
            "note": "Executive function, working memory, and anterior cerebral networks.",
        },
        "Temporal Lobe": {
            "mask": np.maximum(
                gaussian_mask(125, 48, 25, 22),
                gaussian_mask(125, 176, 25, 22)
            ),
            "note": "Lateral neocortex involved in language, semantic memory, and auditory processing.",
        },
        "Parietal Lobe": {
            "mask": np.maximum(
                gaussian_mask(140, 58, 25, 25),
                gaussian_mask(140, 166, 25, 25)
            ),
            "note": "Visuospatial cognition, sensory integration, and parietal association cortex.",
        },
        "Occipital Lobe": {
            "mask": gaussian_mask(180, 112, 22, 38),
            "note": "Primary and secondary visual cortices; typically spared in early Alzheimer's disease.",
        },
        "Hippocampus": {
            "mask": np.maximum(
                gaussian_mask(120, 84, 14, 12),
                gaussian_mask(120, 140, 14, 12)
            ),
            "note": "Bilateral medial temporal lobe structure critical for episodic memory formation and early neurodegeneration.",
        },
        "Entorhinal Cortex": {
            "mask": np.maximum(
                gaussian_mask(105, 90, 12, 10),
                gaussian_mask(105, 134, 12, 10)
            ),
            "note": "Parahippocampal gatekeeper for hippocampal memory routing; earliest site of neurofibrillary tau tangles.",
        },
        "Cingulate Cortex": {
            "mask": gaussian_mask(105, 112, 35, 10),
            "note": "Medial limbic cortex involved in emotional regulation, attention, and default mode network hub.",
        },
        "Precuneus": {
            "mask": gaussian_mask(155, 112, 18, 16),
            "note": "Posteromedial parietal cortex involved in episodic memory retrieval and early default mode network disruption.",
        },
    }

    # Calculate area-weighted density for each region to avoid volume bias
    densities = {}
    for name, cfg in regions_config.items():
        m = cfg["mask"]
        overlap = float((attribution_map * m).sum())
        mask_area = float(m.sum()) + 1e-8
        densities[name] = overlap / mask_area

    total_density = sum(densities.values()) + 1e-8
    normalized_scores = {name: d / total_density for name, d in densities.items()}

    # Rank regions by normalized attribution
    ranked = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)

    region_results: List[Dict[str, Any]] = []
    for name, score in ranked:
        if score >= 0.15:
            level = "High attribution"
        elif score >= 0.10:
            level = "Moderate attribution"
        else:
            level = "Lower attribution"

        region_results.append({
            "region_name": name,
            "attribution_level": level,
            "attribution_score": round(score, 4),
            "clinical_note": regions_config[name]["note"],
        })

    return {
        "regions": region_results,
        "is_estimated": True,
        "methodology": (
            "Estimated image-space anatomical localization based on canonical 2D axial brain template mapping. "
            "Not a 3D clinical volumetric segmentation."
        ),
        "disclaimer": (
            "Brain-region attribution represents model attention/attribution and is not equivalent to "
            "a confirmed anatomical lesion or clinical diagnosis."
        ),
    }


def generate_explanation(
    model,
    image_path: str,
    method: str,
    class_names: list,
    target_class_idx: Optional[int] = None,
) -> dict:
    """
    Generate an XAI heatmap from the actual model and actual uploaded image.
    No stock/sample heatmaps — computed live from CNN activations and gradients.
    """
    import torch

    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((224, 224))
    input_tensor = TRANSFORM(img).unsqueeze(0)
    input_tensor.requires_grad = True

    method_val = method if isinstance(method, str) else method.value

    if method_val in ("gradcam", "gradcam++", "hirescam"):
        return _generate_gradcam(model, input_tensor, img_resized, method_val, target_class_idx)
    elif method_val == "integrated_gradients":
        return _generate_integrated_gradients(model, input_tensor, img_resized, target_class_idx)
    elif method_val == "guided_backprop":
        return _generate_guided_backprop(model, input_tensor, img_resized, target_class_idx)
    else:
        raise ValueError(f"Unsupported explainability method: {method_val}")


def generate_all_explainability(
    model,
    image_path: str,
    target_class_idx: Optional[int] = None,
) -> dict:
    """
    High-efficiency unified explainability generator:
    Preprocesses the scan once, executes Grad-CAM, Grad-CAM++, Integrated Gradients,
    and derives Brain Region Analysis from the primary attribution map.
    Returns both backward-compatible flat keys and structured schema fields.
    """
    import torch
    from loguru import logger

    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((224, 224))
    input_tensor = TRANSFORM(img).unsqueeze(0)
    input_tensor.requires_grad = True

    if target_class_idx is None:
        with torch.no_grad():
            output = model(input_tensor.detach())
            target_class_idx = int(torch.argmax(output, dim=1).item())

    overlays = {}
    structured = {}
    primary_cam_map = None

    # 1. Grad-CAM
    try:
        gcam = _generate_gradcam(model, input_tensor, img_resized, "gradcam", target_class_idx)
        overlays["gradcam_heatmap"] = gcam["heatmap_base64"]
        overlays["gradcam_overlay"] = gcam["overlay_base64"]
        structured["grad_cam"] = {
            "heatmap_base64": gcam["heatmap_base64"],
            "overlay_base64": gcam["overlay_base64"],
            "description": "Highlights coarse convolutional feature activations that contributed most strongly to the predicted class.",
        }
        primary_cam_map = gcam["raw_cam"]
    except Exception as e:
        logger.warning(f"Grad-CAM generation failed: {e}")

    # 2. Grad-CAM++
    try:
        gcam_pp = _generate_gradcam(model, input_tensor, img_resized, "gradcam++", target_class_idx)
        overlays["gradcam_plus_plus_heatmap"] = gcam_pp["heatmap_base64"]
        overlays["gradcam_plus_plus_overlay"] = gcam_pp["overlay_base64"]
        structured["grad_cam_plus_plus"] = {
            "heatmap_base64": gcam_pp["heatmap_base64"],
            "overlay_base64": gcam_pp["overlay_base64"],
            "description": "Provides refined class-discriminative localization using higher-order positive gradient weighting.",
        }
    except Exception as e:
        logger.warning(f"Grad-CAM++ generation failed: {e}")

    # 3. Integrated Gradients
    try:
        ig_res = _generate_integrated_gradients(model, input_tensor, img_resized, target_class_idx)
        overlays["integrated_gradients_heatmap"] = ig_res["heatmap_base64"]
        overlays["integrated_gradients_overlay"] = ig_res["overlay_base64"]
        structured["integrated_gradients"] = {
            "heatmap_base64": ig_res["heatmap_base64"],
            "overlay_base64": ig_res["overlay_base64"],
            "description": "Computes pixel-level path attribution from a zero baseline directly to the input image.",
        }
        if primary_cam_map is None:
            primary_cam_map = ig_res["raw_attribution"]
    except Exception as e:
        logger.warning(f"Integrated Gradients generation failed: {e}")

    # 4. HiResCAM (existing method kept for full compatibility)
    try:
        hires = _generate_gradcam(model, input_tensor, img_resized, "hirescam", target_class_idx)
        overlays["hirescam_heatmap"] = hires["heatmap_base64"]
        overlays["hirescam_overlay"] = hires["overlay_base64"]
        structured["hirescam"] = {
            "heatmap_base64": hires["heatmap_base64"],
            "overlay_base64": hires["overlay_base64"],
            "description": "Retains element-wise spatial feature fidelity across the convolutional layer.",
        }
    except Exception as e:
        logger.warning(f"HiResCAM generation failed: {e}")

    # 5. Brain Region Analysis
    brain_regions = None
    if primary_cam_map is not None:
        try:
            brain_regions = analyze_brain_regions(primary_cam_map)
            structured["brain_regions"] = brain_regions
        except Exception as e:
            logger.warning(f"Brain region analysis failed: {e}")

    return {
        "xai_overlays": overlays,
        "explainability": structured,
        "brain_regions": brain_regions,
    }


def generate_healthy_comparison(scan_path: str) -> dict:
    """
    Compare uploaded scan against a class-averaged healthy brain template.
    The template must be pre-computed from real training data.
    """
    import os

    template_path = os.path.join("ml", "reports", "healthy_template.npy")
    if not os.path.exists(template_path):
        raise FileNotFoundError(
            "Healthy brain template not yet generated. "
            "Run the training pipeline to compute this from real data."
        )

    healthy_template = np.load(template_path)
    scan_img = np.array(Image.open(scan_path).convert("L").resize((224, 224)))
    scan_normalized = scan_img.astype(np.float32) / 255.0

    # Compute difference map
    diff = np.abs(scan_normalized - healthy_template)
    diff_normalized = diff / (diff.max() + 1e-8)

    diff_colored = cv2.applyColorMap(np.uint8(255 * diff_normalized), cv2.COLORMAP_HOT)
    diff_colored = cv2.cvtColor(diff_colored, cv2.COLOR_BGR2RGB)

    return {
        "uploaded_scan_base64": _img_to_base64(scan_img),
        "healthy_template_base64": _img_to_base64((healthy_template * 255).astype(np.uint8)),
        "difference_map_base64": _img_to_base64(diff_colored),
        "description": (
            "Comparison between the uploaded MRI scan and a class-averaged healthy brain template "
            "computed from real training data (Non-Demented class average). "
            "The difference map highlights regions of divergence."
        ),
    }
