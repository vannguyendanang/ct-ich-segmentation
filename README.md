# Intracranial Hemorrhage (ICH) Segmentation using U-Net

A deep learning pipeline for detecting and segmenting intracranial hemorrhage (ICH) regions in CT scan slices using a U-Net model with ResNet34 encoder.

---

## Dataset

This project uses the [PhysioNet CT-ICH dataset](https://physionet.org/content/ct-ich/1.3.1/#files-panel), which contains 82 CT scans from patients with a mean age of 27.8 years, including 36 cases of intracranial hemorrhage across five subtypes: Intraventricular, Intraparenchymal, Subarachnoid, Epidural, and Subdural. Each scan consists of approximately 30 slices at 5mm thickness, annotated by two radiologists who reached consensus on all findings.

---

## Project Structure

```
├── data/
│   ├── image/                  # CT scan slices (PNG, shape 512x512)
│   ├── label/                  # Hemorrhage masks (PNG, shape 512x512)
│   ├── train_split.csv         # Training set split
│   ├── val_split.csv           # Validation set split
│   └── test_split.csv          # Test set split
├── split_data_set.py           # Splits dataset into train/val/test
├── data_augmentation.py        # Augmentation pipeline
├── train_net.py                # Main training script
├── train_net_hyper_finetune.py # Hyperparameter tuning with Optuna
├── best_model.pth              # Saved best model weights
└── README.md
```

---

## Setup

### Requirements

```bash
conda create -n dl_project python=3.8
conda activate dl_project
pip install torch torchvision
pip install segmentation-models-pytorch
pip install albumentations
pip install pandas numpy pillow matplotlib
pip install optuna
```

### Data Preparation

1. Download the CT-ICH dataset from [PhysioNet](https://physionet.org/content/ct-ich/1.3.1/)
2. Run the data preparation script to extract slices:

```bash
python data_preparation.py
```

This will:
- Load CT slices using the brain window (level=40, width=120)
- Clip pixel values to the range 0–255 (HU -20 to 100)
- Save CT slices and binary masks to `data/image/` and `data/label/`

3. Split the dataset into train/val/test (70/20/10):

```bash
python split_data_set.py
```

---

## Training

```bash
python train_net.py
```

### Key Training Parameters

| Parameter | Value |
|-----------|-------|
| Model | U-Net with ResNet34 encoder |
| Input channels | 1 (grayscale) |
| Batch size | 16 |
| Learning rate | 1e-5 |
| Epochs | 200 |
| Optimizer | AdamW (weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR |
| pos_weight cap | 5.0 |
| Early stopping patience | 30 epochs |

### Loss Function

A combined loss of BCE and Dice is used, computed only on positive (hemorrhage) slices:

```
Combined Loss = α × BCE Loss + (1 − α) × Dice Loss
```

Where `α` is a weighting factor controlling the balance between BCE and Dice loss.

### Training Improvements Applied

- **Combined BCE + Dice loss** — BCE alone is dominated by background pixels; Dice loss directly penalizes spatial misalignment of hemorrhage predictions
- **AdamW optimizer** — correctly decouples weight decay from gradient updates, providing stronger regularization for the small dataset
- **CosineAnnealingLR** — smoothly decays learning rate to prevent large weight updates in later epochs
- **Gradient clipping** (`max_norm=1.0`) — prevents exploding gradients caused by the large `pos_weight` penalty
- **Hemorrhage-only augmentation** — augmentation applied only to hemorrhage slices to reduce class imbalance

### Augmentation Pipeline

| Augmentation | Reason |
|-------------|--------|
| Horizontal flip | Brain is roughly symmetric |
| Small rotation (±10°) | Patient positioning varies |
| Brightness/contrast | Scanner variations |
| Zoom/crop | Different patient sizes |

---

## Hyperparameter Tuning

```bash
python train_net_hyper_finetune.py
```

Uses [Optuna](https://optuna.org/) to search for optimal values of:
- Learning rate (`lr`)
- Positive weight (`pos_weight`)
- Loss balancing coefficient (`alpha`)

### Best Parameters Found

| Parameter | Best Value |
|-----------|-----------|
| lr | 1.56e-03 |
| pos_weight | 8.72 |
| alpha | 0.627 |

---

## Results

### Validation Performance

| Stage | Best Dice |
|-------|-----------|
| Baseline (BCE only) | 0.341 |
| After improvements | 0.444 |

### Test Performance

| Metric | Score |
|--------|-------|
| Dice score | 0.321 |
| Sensitivity | 0.319 |

A Dice score of 0.321 means approximately 32% overlap between predicted and actual hemorrhage pixels. Sensitivity of 0.319 means the model correctly identifies 31.9% of actual hemorrhage pixels. Scores above 0.70 are typically considered clinically useful for medical image segmentation.

---

## Evaluation Metrics

**Dice Score** measures overlap between predicted and actual hemorrhage regions:

```
Dice = 2 × |A ∩ B| / (|A| + |B|)
```

**Sensitivity** measures how well the model catches actual hemorrhage pixels:

```
Sensitivity = TP / (TP + FN)
```

Where TP = correctly predicted hemorrhage pixels, FN = missed hemorrhage pixels.

Normal accuracy is not used because a model predicting all background achieves ~95% accuracy while missing all hemorrhage regions entirely.

---

## Running on HPC (Cheaha)

```bash
sbatch train_job.sh
sbatch train_job_hyper_tuning.sh
```

Example SLURM script:

```bash
#!/bin/bash
#SBATCH --job-name=unet_train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --partition=pascalnodes
#SBATCH --output=train_%j.log

module load Anaconda3
conda activate dl_project
cd ~/Data/physionet.org/files/ct-ich/material/
python train_net.py
```

---

## Limitations and Future Work

- The low Dice score is expected given severe class imbalance (11.5% hemorrhage slices vs 88.5% non-hemorrhage)
- Increasing augmentations from 3 to 5 per hemorrhage slice may improve generalization
- Switching to a deeper encoder (ResNet50) may capture more subtle hemorrhage features
- A larger dataset would significantly improve model generalization
