# USDNet Full Training Guide

## 📋 Training Pipeline Overview

USDNet adopts a **two-stage training** strategy:

1. **Pre-training stage**: only train DINO feature distillation so that the backbone learns 2D features  
2. **Fine-tuning stage**: load the pre-trained weights and train semantic segmentation (optionally keep distillation enabled)

---

## 🐳 Start the Docker Container

```bash
# Stop and remove old container
docker stop usdnet_container
docker rm usdnet_container

# Start a new container
docker run -it   --name usdnet_container   --gpus all   --privileged   -v /home/shaolongshi/data/pointnet.pytorch:/workspace/project   -v /home/shaolongshi/data/pointnet.pytorch/utils/2test:/workspace/dataset   -v /media/shaolongshi/Elements:/workspace/data_elements   -v /home/shaolongshi/output:/workspace/output   -v /mnt/nas/scannetpp_new/data:/mnt/nas/scannetpp_new/data   usdnet:latest /bin/bash

# Go to working directory and install dependencies
cd /workspace/project/utils/2test
pip install zarr timm scikit-learn
```

---

## 🚀 Full Training Steps

### Step 1: Generate distillation data (already done)

```bash
python batch_ditr_paper_strict_zarr.py
```

---

### Step 2: Pre-training (DINO distillation)

```bash
python train_usdnet_complete.py   --stage pretrain   --zarr_root /workspace/data_elements/newz   --batch_size 2   --max_epochs 50   --learning_rate 1e-3   --device cuda:0   --save_dir ./checkpoints
```

**Outputs:**

- Models are saved in `./checkpoints/pretrain/`
- Best model: `./checkpoints/pretrain/ckpt_best.pt`

---

### Step 3: Fine-tuning (semantic segmentation)

**Use pre-trained weights (recommended)**

```bash
python train_usdnet_complete.py   --stage finetune   --zarr_root /workspace/data_elements/newz   --pretrain_checkpoint ./checkpoints/pretrain/ckpt_best.pt   --batch_size 2   --max_epochs 100   --learning_rate 1e-3   --device cuda:0   --save_dir ./checkpoints   --use_val
```

**Outputs:**

- Models are saved in `./checkpoints/finetune/`
- Best model: `./checkpoints/finetune/ckpt_best.pt`

---

### Step 4: Visualization

**Object-wise visualization (default)**

```bash
# Basic usage - generate a separate HTML page for each object class
python visualize_usdnet_predictions.py   --checkpoint ./checkpoints/finetune/ckpt_best.pt   --zarr_root /workspace/data_elements/newz   --output_dir ./visualizations

# Specify a scene and inference chunk size
python visualize_usdnet_predictions.py   --checkpoint ./checkpoints/finetune/ckpt_best.pt   --zarr_root /workspace/data_elements/newz   --scene_id 00777c41d4   --infer_chunk_size 100000   --max_points_per_object 50000   --output_dir ./visualizations
```

**Argument notes:**

- `--infer_chunk_size`: chunk size during inference (controls GPU memory usage, default 100k)
- `--max_points_per_object`: maximum number of points per object; if exceeded, the object is automatically split spatially (default 50k)
- `--max_points`: limit on total downsampled points (optional, useful for quick preview)

**Outputs:**

- Index page: `{scene_id}_objects_index.html`
- Per-object pages: `{scene_id}_{class_name}.html`
- Large objects are automatically split into multiple parts
