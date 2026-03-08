"""
app.py – Streamlit demo for BoneVision Fracture Classifier.

Run:
    streamlit run app.py

If you trained on Colab, download the checkpoint first:
    - From Google Drive → My Drive/KBG_Results/checkpoints/best_model.pth
    - Place it at: checkpoints/best_model.pth  (relative to this file)
"""

import os
import sys
import io
import yaml
import numpy as np
import torch
import torch.nn.functional as F
import streamlit as st
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── Make sure local imports work ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from model import FractureClassifier, SoftVotingEnsemble, GradCAMWrapper

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BoneVision – Fracture Classifier",
    page_icon="🦴",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES = ["fractured", "not fractured"]
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_inference_transform():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


@st.cache_resource(show_spinner="Loading model…")
def load_model(checkpoint_path: str):
    """Load the trained model from a checkpoint."""
    device = get_device()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Detect architecture from checkpoint keys
    # Support multiple checkpoint formats
    if "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # Check if ensemble (has keys like "models.0.backbone…") or single model
    is_ensemble = any(k.startswith("models.") for k in state_dict.keys())

    if is_ensemble:
        # Discover how many sub-models
        model_indices = set()
        for k in state_dict.keys():
            if k.startswith("models."):
                idx = int(k.split(".")[1])
                model_indices.add(idx)
        n_models = len(model_indices)

        # Try to read config for architecture names; fallback to defaults
        cfg_path = os.path.join(SCRIPT_DIR, "config.yaml")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            arch_names = cfg["model"]["ensemble"]["models"][:n_models]
            weights = cfg["model"]["ensemble"]["weights"][:n_models]
        else:
            arch_names = ["vit_small_patch16_224", "efficientnet_b0", "convnext_tiny"][:n_models]
            weights = [0.5, 0.25, 0.25][:n_models]

        models = [
            FractureClassifier(name, num_classes=len(CLASS_NAMES), pretrained=False)
            for name in arch_names
        ]
        model = SoftVotingEnsemble(models, weights=weights)
    else:
        # Single model — try to detect architecture name
        cfg_path = os.path.join(SCRIPT_DIR, "config.yaml")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            arch = cfg["model"]["architecture"]
        else:
            arch = "vit_small_patch16_224"
        model = FractureClassifier(arch, num_classes=len(CLASS_NAMES), pretrained=False)

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model, device


def get_gradcam_target_layer(model):
    """Pick the best target layer for GradCAM based on model type."""
    if isinstance(model, SoftVotingEnsemble):
        # Use first model in ensemble
        inner = model.models[0]
    else:
        inner = model

    backbone = inner.backbone

    # Try common layer names
    for attr in ["norm", "layer4", "features", "stages"]:
        if hasattr(backbone, attr):
            layer = getattr(backbone, attr)
            if isinstance(layer, (torch.nn.Sequential, torch.nn.ModuleList)):
                return layer[-1]
            return layer

    # Fallback — last module
    children = list(backbone.children())
    return children[-1] if children else backbone


def generate_gradcam(model, image_tensor, device, class_idx):
    """Generate GradCAM heatmap. Returns heatmap as numpy array (H, W)."""
    target_layer = get_gradcam_target_layer(model)
    cam = GradCAMWrapper(model, target_layer)
    try:
        heatmap = cam(image_tensor.unsqueeze(0).to(device), class_idx=class_idx)
        if heatmap.ndim == 1:
            # ViT: reshape token sequence to 2D (exclude CLS token)
            n_tokens = heatmap.shape[0]
            if n_tokens == 197:  # 14*14 + 1 (CLS)
                heatmap = heatmap[1:].reshape(14, 14)
            else:
                side = int(np.sqrt(n_tokens))
                if side * side == n_tokens:
                    heatmap = heatmap.reshape(side, side)
                else:
                    heatmap = heatmap[1:n_tokens]
                    side = int(np.sqrt(n_tokens - 1))
                    heatmap = heatmap[:side * side].reshape(side, side)
        heatmap = heatmap.numpy()
        heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
        return heatmap
    except Exception as e:
        st.warning(f"GradCAM failed: {e}")
        return None


def overlay_heatmap(original_img, heatmap, alpha=0.5):
    """Overlay GradCAM heatmap on original image."""
    img = np.array(original_img.resize((IMG_SIZE, IMG_SIZE)))
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return overlay


def predict(model, image: Image.Image, device):
    """Run inference on a PIL Image, return class probs."""
    transform = build_inference_transform()
    img_np = np.array(image.convert("RGB"))
    transformed = transform(image=img_np)
    tensor = transformed["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()

    return probs, tensor.squeeze(0)


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── Header ──
    st.markdown(
        """
        <div style='text-align: center; padding: 1rem 0;'>
            <h1>🦴 BoneVision – Fracture Detector</h1>
            <p style='font-size: 1.05rem; color: grey; margin-top: -0.6rem;'>
                Upload a bone X-ray, get a fracture / no-fracture verdict with GradCAM explainability.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Sidebar ──
    with st.sidebar:
        st.header("⚙️ Settings")

        # Checkpoint selector
        default_ckpt = os.path.join(SCRIPT_DIR, "checkpoints", "best_model.pth")
        ckpt_path = st.text_input(
            "Checkpoint path",
            value=default_ckpt,
            help="Path to the trained model checkpoint (.pth file)",
        )

        show_gradcam = st.checkbox("Show GradCAM", value=True)
        gradcam_alpha = st.slider("GradCAM overlay opacity", 0.1, 0.9, 0.5, 0.05)

    # ── Model loading ──
    if not os.path.exists(ckpt_path):
        st.warning(
            f"⚠️ Checkpoint not found at `{ckpt_path}`.  \n"
            "Please download `best_model.pth` from Google Drive → `My Drive/KBG_Results/checkpoints/` "
            "and place it in the `checkpoints/` folder."
        )
        st.info(
            "💡 **Tip:** You can also enter the full path to any `.pth` file in the sidebar."
        )
        st.stop()

    model, device = load_model(ckpt_path)
    st.sidebar.success(f"Model ready on **{device}**")

    # ── Upload ──
    col_upload, col_result = st.columns([1, 1])

    with col_upload:
        st.subheader("📤 Upload X-Ray")
        uploaded = st.file_uploader(
            "Choose an X-ray image",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            help="Upload a bone X-ray image for fracture classification",
        )

        # Sample images toggle
        use_sample = st.checkbox("Use a sample image instead")
        if use_sample:
            sample_dir = os.path.join(SCRIPT_DIR, "..", "..", "data", "clean_split", "test")
            if os.path.exists(sample_dir):
                frac_dir = os.path.join(sample_dir, "fractured")
                nofrac_dir = os.path.join(sample_dir, "not fractured")
                sample_class = st.radio("Sample class:", ["fractured", "not fractured"])
                chosen_dir = frac_dir if sample_class == "fractured" else nofrac_dir
                if os.path.exists(chosen_dir):
                    files = sorted([f for f in os.listdir(chosen_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                    if files:
                        chosen_file = st.selectbox("Choose image:", files[:20])
                        sample_path = os.path.join(chosen_dir, chosen_file)
                        uploaded = open(sample_path, "rb")
                    else:
                        st.info("No sample images found.")
            else:
                st.info("Sample data directory not found. Upload an image instead.")

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")

        with col_upload:
            st.image(image, caption="Input X-Ray", width="stretch")

        # ── Predict ──
        with st.spinner("🔍 Analyzing…"):
            probs, img_tensor = predict(model, image, device)

        pred_idx = int(np.argmax(probs))
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx]) * 100

        with col_result:
            st.subheader("🩻 Prediction Result")

            # Color-coded result
            if pred_class == "fractured":
                st.error(f"### 🚨 **FRACTURED** — {confidence:.1f}% confidence")
            else:
                st.success(f"### ✅ **NOT FRACTURED** — {confidence:.1f}% confidence")

            # Probability bars
            st.markdown("**Class probabilities**")
            for i, cls_name in enumerate(CLASS_NAMES):
                pct = float(probs[i]) * 100
                st.progress(float(probs[i]), text=f"{cls_name}: {pct:.1f}%")

            # GradCAM
            if show_gradcam:
                st.markdown("---")
                st.subheader("🔥 GradCAM — Model Attention")
                with st.spinner("Generating GradCAM…"):
                    heatmap = generate_gradcam(model, img_tensor, device, pred_idx)
                if heatmap is not None:
                    overlay = overlay_heatmap(image, heatmap, alpha=gradcam_alpha)
                    cam_col1, cam_col2 = st.columns(2)
                    with cam_col1:
                        st.image(image.resize((IMG_SIZE, IMG_SIZE)), caption="Original", width="stretch")
                    with cam_col2:
                        st.image(overlay, caption="GradCAM Overlay", width="stretch")
                    st.caption(
                        "GradCAM highlights the regions the model focused on. "
                        "Red = high attention, Blue = low attention."
                    )
                else:
                    st.info("GradCAM visualization unavailable for this model architecture.")

    else:
        with col_result:
            st.info("👈 Upload an X-ray image or select a sample to get started.")


if __name__ == "__main__":
    main()
