# Bone Fracture Classification
**Kamand Bioengineering Group Hackathon 2025 | IIT Mandi**

## Data Leakage Check & Clean Split
Before training, detect and fix any cross-split duplicate leakage:

```bash
# 1. Check for duplicates (exact SHA1 + perceptual hash)
python scripts/dedupe_report.py --root ../../data/Bone_Fracture_Binary_Classification --output leak_report.csv

# 2. Build leak-free 70/10/20 stratified splits (hardlinks, no extra disk)
python scripts/clean_resplit.py \
  --source ../../data/Bone_Fracture_Binary_Classification \
  --dest   ../../data/clean_split \
  --val 0.10 --test 0.20 --mode link --force
```
Config is already pointed at `data/clean_split/`. To revert to the raw dataset,
swap the commented paths in `config.yaml`.

## Architecture
A **soft-voting ensemble** of three ImageNet-pretrained backbones:

| Backbone | Role | Weight |
|---|---|---|
| ViT-B/16 (`vit_base_patch16_224`) | Primary – global attention | 50% |
| EfficientNet-B3 | Compact CNN features | 25% |
| ConvNeXt-Small | Hierarchical local features | 25% |

Key design choices:
- All models fine-tuned from **ImageNet** (no fracture-specific pre-training, per rules)
- **Label smoothing** cross-entropy + **MixUp** augmentation for better generalization
- **Weighted class sampler** to handle imbalanced fracture categories
- **GradCAM** attention maps for radiologist-interpretable explainability
- 5-fold stratified cross-validation for reliable generalization estimates
- Apple Silicon (MPS) & CUDA auto-detection

## Dataset
Binary classification: **fractured** vs **not fractured**
- Train: ~9,246 images | Val: ~829 images | Test: ~506 images
- Format: PNG/JPEG, variable resolution → standardized to 224×224

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Navigate to the project directory
cd KBG_Hackathon_Submission/bone_fracture_classifier

# 3. Run leakage check + clean resplit (one-time)
python scripts/dedupe_report.py --root ../../data/Bone_Fracture_Binary_Classification
python scripts/clean_resplit.py --source ../../data/Bone_Fracture_Binary_Classification \
  --dest ../../data/clean_split --val 0.10 --test 0.20 --force

# 4. Train (single run)
python train.py --config config.yaml

# 5. Train with 5-fold CV
python train.py --config config.yaml --cv

# 6. Evaluate on test set (generates all CSVs + plots)
python evaluate.py --config config.yaml
```

## Repository Structure
```
bone_fracture_classifier/
├── config.yaml                  # All hyperparameters
├── requirements.txt             # Pinned dependencies
├── data_loader.py               # Dataset, augmentation, CV splits
├── model.py                     # ViT / EfficientNet / ConvNeXt ensemble
├── train.py                     # Training loop + epoch CSV logger
├── evaluate.py                  # Test metrics, confusion matrix, GradCAM
├── scripts/
│   ├── dedupe_report.py         # SHA1 + pHash cross-split leakage detector
│   └── clean_resplit.py         # Dedupe & stratified re-split builder
├── TEAM.txt                     # Team information
├── README.md                    # This file
├── checkpoints/
│   └── best_model.pth           # Best checkpoint (val macro-F1)
└── results/
    ├── final_results.csv            # Submission metric sheet
    ├── model_performance_analysis.csv  # Epoch-by-epoch log
    ├── confusion_matrix.png
    ├── roc_curves.png
    └── gradcam_samples.png
```

## Augmentation Strategy
- Horizontal flip, rotation (±15°), affine transforms
- CLAHE contrast enhancement (X-ray specific)
- Grid distortion (simulates X-ray artifacts)
- MixUp (α=0.2) for regularization
- Coarse dropout (random erasing)
- ImageNet normalization

## Hyperparameters
| Parameter | Value |
|---|---|
| Image size | 224 × 224 |
| Batch size (effective) | 64 (32 × 2 accumulation) |
| Optimizer | AdamW (lr=1e-4, wd=1e-2) |
| Scheduler | Cosine + 3-epoch warmup |
| Loss | Label Smoothing CE (ε=0.1) |
| Epochs | 30 (early stop patience=7) |
| Mixed precision | fp16 (CUDA only) |

## Contact
Team members listed in TEAM.txt
