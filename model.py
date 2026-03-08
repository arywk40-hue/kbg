"""
model.py – Model architectures for bone fracture classification.

Architectures
─────────────
  • ViT (Vision Transformer)   – vit_base_patch16_224  [primary]
  • EfficientNet-B3            – efficientnet_b3        [ensemble member]
  • ConvNeXt-Small             – convnext_small         [ensemble member]
  • SoftVotingEnsemble         – weighted average of above

All backbone weights are loaded from ImageNet (timm library).
Medical/fracture pre-trained weights are NOT used (per hackathon rules).

GradCAM-compatible wrapper is provided for explainability (Slide 7).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import yaml
from typing import List, Optional, Dict


# ─────────────────────────────────────────────────────────────────────────────
# Single backbone wrapper
# ─────────────────────────────────────────────────────────────────────────────

class FractureClassifier(nn.Module):
    """
    Generic wrapper around any timm backbone.

    Parameters
    ----------
    model_name   : timm architecture name
    num_classes  : number of fracture categories
    pretrained   : load ImageNet weights
    dropout      : dropout before final classifier head
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        try:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,          # remove default head
                drop_rate=dropout,
            )
        except Exception as exc:
            # Offline / cache-miss fallback: build without pretrained weights
            print(f"[Model] Warning: failed to load pretrained weights for {model_name} ({exc}). "
                  f"Falling back to random init.")
            self.backbone = timm.create_model(
                model_name,
                pretrained=False,
                num_classes=0,
                drop_rate=dropout,
            )
        feature_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )
        self.model_name = model_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ─────────────────────────────────────────────────────────────────────────────
# Soft Voting Ensemble
# ─────────────────────────────────────────────────────────────────────────────

class SoftVotingEnsemble(nn.Module):
    """
    Weighted soft-voting ensemble of multiple FractureClassifier models.

    Parameters
    ----------
    models  : list of FractureClassifier instances
    weights : per-model weights (will be normalized); None = uniform
    """

    def __init__(
        self,
        models: List[FractureClassifier],
        weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.models = nn.ModuleList(models)
        if weights is None:
            weights = [1.0] * len(models)
        w = torch.tensor(weights, dtype=torch.float32)
        self.register_buffer("weights", w / w.sum())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Weighted average of individual logits (not softmaxed).
        # This keeps output in logit-space so standard CE / label-smoothing CE
        # losses work correctly without double-log issues.
        logits = torch.stack(
            [m(x) for m in self.models], dim=0
        )  # (num_models, B, C)
        weighted = (logits * self.weights[:, None, None]).sum(dim=0)  # (B, C)
        return weighted

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            probs = torch.stack(
                [F.softmax(m(x), dim=-1) for m in self.models], dim=0
            )
            return (probs * self.weights[:, None, None]).sum(dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# Label Smoothing Cross Entropy Loss
# ─────────────────────────────────────────────────────────────────────────────

class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy with label smoothing for better calibration."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_cls = logits.size(-1)
        log_prob = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            smooth_targets = torch.full_like(log_prob, self.smoothing / (n_cls - 1))
            smooth_targets.scatter_(-1, targets.unsqueeze(-1), 1.0 - self.smoothing)
        loss = -(smooth_targets * log_prob).sum(dim=-1).mean()
        return loss


class MixUpLoss(nn.Module):
    """Wraps any criterion to support MixUp (labels_a, labels_b, lam) tuples."""

    def __init__(self, criterion: nn.Module):
        super().__init__()
        self.criterion = criterion

    def forward(self, logits: torch.Tensor, targets) -> torch.Tensor:
        if isinstance(targets, tuple):
            labels_a, labels_b, lam = targets
            return lam * self.criterion(logits, labels_a) + \
                   (1 - lam) * self.criterion(logits, labels_b)
        return self.criterion(logits, targets)


# ─────────────────────────────────────────────────────────────────────────────
# GradCAM wrapper (for Slide 7 explainability)
# ─────────────────────────────────────────────────────────────────────────────

class GradCAMWrapper:
    """
    Minimal GradCAM for any model. Hooks the last convolutional / attention
    layer to produce class activation maps.

    Usage
    -----
        cam = GradCAMWrapper(model, target_layer)
        heatmap = cam(image_tensor, class_idx)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self._activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, x: torch.Tensor, class_idx: Optional[int] = None) -> torch.Tensor:
        self.model.eval()
        self.model.zero_grad()
        x = x.requires_grad_(True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=-1).item()
        self.model.zero_grad()
        logits[0, class_idx].backward(retain_graph=True)

        grads = self._gradients
        acts  = self._activations

        # Handle different tensor shapes (CNN 4D vs ViT 3D)
        if grads.ndim == 4:
            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = (weights * acts).sum(dim=1).squeeze()
        elif grads.ndim == 3:
            # (B, num_tokens, dim) -> average over dim
            weights = grads.mean(dim=-1, keepdim=True)
            cam = (weights * acts).sum(dim=-1).squeeze()
            # Remove CLS token if present
            if cam.ndim == 1:
                pass  # already squeezed
        else:
            weights = grads.mean(dim=-1, keepdim=True)
            cam = (weights * acts).sum(dim=-1).squeeze()

        cam = F.relu(cam)
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.cpu().detach()


# ─────────────────────────────────────────────────────────────────────────────
# Factory function
# ─────────────────────────────────────────────────────────────────────────────

def build_model(cfg: dict, num_classes: int) -> nn.Module:
    """
    Build model from config. Returns either a single FractureClassifier
    or a SoftVotingEnsemble based on cfg['model']['ensemble']['enabled'].
    """
    m_cfg = cfg["model"]

    if m_cfg["ensemble"]["enabled"]:
        models = [
            FractureClassifier(
                model_name=name,
                num_classes=num_classes,
                pretrained=m_cfg["pretrained"],
                dropout=m_cfg["dropout"],
            )
            for name in m_cfg["ensemble"]["models"]
        ]
        return SoftVotingEnsemble(models, weights=m_cfg["ensemble"]["weights"])

    return FractureClassifier(
        model_name=m_cfg["architecture"],
        num_classes=num_classes,
        pretrained=m_cfg["pretrained"],
        dropout=m_cfg["dropout"],
    )


def build_criterion(cfg: dict) -> nn.Module:
    """Return the appropriate loss function wrapped for MixUp support."""
    loss_cfg = cfg["training"]["loss"]
    if loss_cfg["name"] == "label_smoothing_cross_entropy":
        base = LabelSmoothingCrossEntropy(smoothing=loss_cfg["smoothing"])
    else:
        base = nn.CrossEntropyLoss()
    return MixUpLoss(base)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    num_classes = len(cfg["data"]["classes"])
    model = build_model(cfg, num_classes)
    params = count_parameters(model)
    print(f"Model     : {type(model).__name__}")
    print(f"Params    : {params['total']:,} total | {params['trainable']:,} trainable")
    x = torch.randn(2, 3, cfg["data"]["img_size"], cfg["data"]["img_size"])
    out = model(x)
    print(f"Output    : {out.shape}")  # (2, num_classes)
