Here is the complete, cohesive `README.md` content.

I have integrated your visual modification requests (changing "Installation" to "**Environment Configuration**", updating the **Citation** status to "Under Submission") and refined the technical details to strictly match your provided code (e.g., the specific temperature parsing logic and Pre-activation GNN architecture).

You can copy the code block below directly.

```markdown
# Hydrogel GNN Temperature Predictor

A Graph Neural Network (GNN) framework for predicting the gelation temperature of hydrogel materials based on molecular structure.

## Overview

This project implements multiple GNN architectures to predict the Lower Critical Solution Temperature (LCST) or Upper Critical Solution Temperature (UCST) of hydrogel polymers from their SMILES representations. The model fuses molecular graph features with compositional information (ligand proportions) for accurate temperature prediction.

## Features

- **Multiple GNN Architectures**: MPNN, GCN, GIN, GAT, GATv2, PAN, Transformer.
- **Robust Data Processing**: Automatically handles complex temperature formats (ranges like `30-35`, inequalities like `>30`).
- **Pre-activation ResNet**: Uses modern pre-activation residual connections for deeper GNN training stability.
- **On-the-fly Augmentation**: Randomizes SMILES strings during training to improve generalization.
- **Cross-Validation**: K-fold cross-validation with distribution-aware split selection.

## Environment Configuration

### 1. Basic Setup

First, set up a virtual environment and install core dependencies:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install core dependencies (PyTorch, RDKit, Pandas, etc.)
pip install -r requirements.txt

```

###2. PyTorch Geometric Installation**Crucial Step:** PyTorch Geometric installation depends strictly on your CUDA version and OS. The commands below are examples for **CUDA 11.8**.

If you use a different CUDA version (e.g., 12.x) or CPU only, please refer to the [official PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

```bash
# For CUDA 11.8 (Example)
pip install torch-geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f [https://data.pyg.org/whl/torch-2.0.0+cu118.html](https://data.pyg.org/whl/torch-2.0.0+cu118.html)

# For CPU Only
# pip install torch-geometric
# pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f [https://data.pyg.org/whl/torch-2.0.0+cpu.html](https://data.pyg.org/whl/torch-2.0.0+cpu.html)

```

##Data Format**Note: The original dataset is not publicly available. Users need to prepare their own data.**

For detailed instructions on preparing your dataset, see [DATA_TEMPLATE.md](https://www.google.com/search?q=DATA_TEMPLATE.md).

The input data should be an Excel file with the following key columns:

| Column | Description |
| --- | --- |
| `SMILES` | Main polymer SMILES string |
| `SMILES_prop` | Proportion of main polymer (0-1) |
| `ligand1...4` | Ligand SMILES strings |
| `ligand1...4_prop` | Ligand proportions |
| `temp_raw` | Target temperature (°C) |

**Smart Parsing Capabilities:**
The preprocessing script (`create_dataset_splits.py`) automatically handles various temperature formats found in literature:

* **Ranges**: `30-35` → converted to mean `32.5`
* **Inequalities**: `>30`, `≥32`, `~35` → symbols stripped, numeric value retained
* **Standard**: `32.5` → kept as is

##Usage###1. Create Dataset SplitsCreate multiple high-quality train/validation/test splits. The script minimizes Wasserstein distance to ensure similar temperature distributions across splits.

```bash
python create_dataset_splits.py \
    --data_path your_data.xlsx \
    --output_dir ./dataset_splits \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --n_splits_to_save 10

```

###2. Train ModelsRun the fused GNN training pipeline.

```bash
python train_fused.py \
    --splits_root_dir ./dataset_splits \
    --model_type MPNN \
    --epochs 400 \
    --batch_size 16 \
    --lr 0.0005 \
    --loss_func Huber \
    --scheduler Cosine

```

**Key Parameters:**

* `--model_type`: `MPNN`, `GCN`, `GIN`, `GAT`, `GATv2`, `PAN`, `Transformer`
* `--loss_func`: `Huber` (default, robust to outliers) or `MSE`.
* `--scheduler`: Learning rate scheduler (`Cosine`, `Step`, `Plateau`).
* `--folds`: Specific folds to run (e.g., `--folds 1 2`).

##Model ArchitectureThe model utilizes a **Pre-activation Residual** architecture for improved gradient flow:

```
Input: Molecular Graph (atoms + bonds) + Proportions Vector
    │
    ├── Atom Embedding Layer + Linear Encoder
    │       │
    │       ▼
    ├── GNN Layers (Pre-activation Residual Blocks)
    │   ├── ┌── BatchNorm → Activation → GNN Conv ──┐
    │   └── ┴─────────── Residual (+) ──────────────┘
    │       │
    │       ▼
    ├── Set2Set Pooling (graph-level representation)
    │       │
    │       ▼
    └── Proportion MLP ──────┐
                             │
                             ▼
                    Concatenation
                             │
                             ▼
                    Final MLP → Temperature Prediction

```

##Output StructureAfter training, outputs are saved in `training_output/`:

```
training_output/
└── {MODEL}_lr{LR}_bs{BS}_{TIMESTAMP}/
    ├── logs/
    │   ├── training_config.json
    │   ├── fold_results.csv        # Metrics for each fold
    │   └── final_summary.csv       # Aggregated mean ± std
    ├── models/
    │   └── fold_{N}/
    │       ├── model_fold{N}.pth
    │       ├── scaler_fold{N}.joblib
    │       └── feature_stats_fold{N}.json
    └── plot_data/
        └── fold_{N}/
            ├── loss_curve.png
            └── prediction_scatter.png

```

##Evaluation Metrics* **Primary**: MAE, RMSE, R²
* **Correlations**: Pearson r, Spearman ρ, Kendall τ

##CitationThis work is currently **under submission**.

If you use this code in your research, please check back for updated citation information upon publication.

##LicenseMIT License

##ContactFor questions or issues, please open an issue on GitHub.

```

```
