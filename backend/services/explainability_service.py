"""
NeuroAssist AI v2 — Explainability Service
Generates XAI visualizations from REAL model activations on REAL uploaded scans.
Never returns stock/sample heatmap images.
"""

import numpy as np
import base64
import io
from PIL import Image
from torchvision import transforms

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _img_to_base64(img_array: np.ndarray) -> str:
    """Convert numpy array to base64-encoded PNG."""
    if img_array.dtype != np.uint8:
        img_array = (img_array * 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def generate_explanation(model, image_path: str, method: str, class_names: list) -> dict:
    """
    Generate an XAI heatmap from the actual model and actual uploaded image.
    No stock/sample heatmaps — this is computed live.
    """
    import torch

    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((224, 224))
    input_tensor = TRANSFORM(img).unsqueeze(0)
    input_tensor.requires_grad = True

    method_val = method if isinstance(method, str) else method.value

    if method_val in ("gradcam", "gradcam++", "hirescam"):
        return _generate_gradcam(model, input_tensor, img_resized, method_val)
    elif method_val == "integrated_gradients":
        return _generate_integrated_gradients(model, input_tensor, img_resized)
    elif method_val == "guided_backprop":
        return _generate_guided_backprop(model, input_tensor, img_resized)
    else:
        raise ValueError(f"Unsupported explainability method: {method_val}")


def _generate_gradcam(model, input_tensor, original_img, method: str) -> dict:
    """Generate Grad-CAM / Grad-CAM++ / HiResCAM via pytorch-grad-cam."""
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, HiResCAM
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
    grayscale_cam = cam(input_tensor=input_tensor.detach())[0]

    # Create overlay on original image
    img_array = np.array(original_img).astype(np.float32) / 255.0
    overlay = show_cam_on_image(img_array, grayscale_cam, use_rgb=True)

    # Heatmap only (as color map)
    import cv2
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    return {
        "heatmap_base64": _img_to_base64(heatmap_colored),
        "overlay_base64": _img_to_base64(overlay),
    }


def _generate_integrated_gradients(model, input_tensor, original_img) -> dict:
    """Generate Integrated Gradients via Captum."""
    from captum.attr import IntegratedGradients
    import torch

    ig = IntegratedGradients(model)
    baseline = torch.zeros_like(input_tensor)

    with torch.no_grad():
        output = model(input_tensor.detach())
        predicted_class = torch.argmax(output, dim=1).item()

    attributions = ig.attribute(
        input_tensor, baselines=baseline, target=predicted_class, n_steps=50
    )

    # Convert attributions to heatmap
    attr_np = attributions.squeeze().detach().numpy()
    attr_magnitude = np.abs(attr_np).sum(axis=0)  # Sum across channels
    attr_normalized = attr_magnitude / (attr_magnitude.max() + 1e-8)

    import cv2
    heatmap = cv2.applyColorMap(np.uint8(255 * attr_normalized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay
    img_array = np.array(original_img).astype(np.float32) / 255.0
    overlay_array = img_array * 0.5 + (heatmap.astype(np.float32) / 255.0) * 0.5

    return {
        "heatmap_base64": _img_to_base64(heatmap),
        "overlay_base64": _img_to_base64(overlay_array),
    }


def _generate_guided_backprop(model, input_tensor, original_img) -> dict:
    """Generate Guided Backpropagation via Captum."""
    from captum.attr import GuidedBackprop
    import torch

    gbp = GuidedBackprop(model)

    with torch.no_grad():
        output = model(input_tensor.detach())
        predicted_class = torch.argmax(output, dim=1).item()

    attributions = gbp.attribute(input_tensor, target=predicted_class)

    attr_np = attributions.squeeze().detach().numpy()
    attr_magnitude = np.abs(attr_np).sum(axis=0)
    attr_normalized = attr_magnitude / (attr_magnitude.max() + 1e-8)

    import cv2
    heatmap = cv2.applyColorMap(np.uint8(255 * attr_normalized), cv2.COLORMAP_INFERNO)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    img_array = np.array(original_img).astype(np.float32) / 255.0
    overlay_array = img_array * 0.5 + (heatmap.astype(np.float32) / 255.0) * 0.5

    return {
        "heatmap_base64": _img_to_base64(heatmap),
        "overlay_base64": _img_to_base64(overlay_array),
    }


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

    import cv2
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
