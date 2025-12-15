# Hydrogel GNN Temperature Predictor

A Graph Neural Network (GNN) framework for predicting the gelation temperature of hydrogel materials based on molecular structure.

## Overview

This project implements multiple GNN architectures to predict the Lower Critical Solution Temperature (LCST) or Upper Critical Solution Temperature (UCST) of hydrogel polymers from their SMILES representations. The model fuses molecular graph features with compositional information (ligand proportions) for accurate temperature prediction.

## Features

- **Multiple GNN Architectures**: MPNN, GCN, GIN, GAT, GATv2, PAN, Transformer
- **Molecular Feature Extraction**: Atom-level and bond-level features from RDKit
- **Data Augmentation**: SMILES augmentation for improved generalization
- **Cross-Validation**: K-fold cross-validation with quality-based split selection

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### PyTorch Geometric Installation

PyTorch Geometric requires specific installation based on your CUDA version:

```bash
# For CUDA 11.8
pip install torch-geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# For CPU only
pip install torch-geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

## Data Format

**Note: The original dataset is not publicly available. Users need to prepare their own data.**

For detailed instructions on preparing your dataset, see [DATA_TEMPLATE.md](DATA_TEMPLATE.md).

The input data should be an Excel file (e.g., `SMILES.xlsx`) with the following columns:

| Column | Description |
|--------|-------------|
| `SMILES` | Main polymer SMILES string |
| `SMILES_prop` | Proportion of main polymer (0-1) |
| `ligand1` | First ligand SMILES |
| `ligand1_prop` | First ligand proportion |
| `ligand2` | Second ligand SMILES |
| `ligand2_prop` | Second ligand proportion |
| `ligand3` | Third ligand SMILES |
| `ligand3_prop` | Third ligand proportion |
| `ligand4` | Fourth ligand SMILES |
| `ligand4_prop` | Fourth ligand proportion |
| `temp_raw` | Target temperature (°C) |

## Usage

### 1. Prepare Your Data

Prepare your own dataset in Excel format with the required columns (see Data Format section above).

### 2. Create Dataset Splits

Create multiple high-quality train/validation/test splits with distribution optimization:

```bash
python create_dataset_splits.py \
    --data_path your_data.xlsx \
    --output_dir ./dataset_splits \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --n_splits_to_save 10 \
    --n_candidates 50
```

**Parameters:**
- `--n_splits_to_save`: Number of best splits to save (default: 10)
- `--n_candidates`: Number of candidate splits to evaluate (default: 50)
- Split quality is evaluated using Wasserstein distance to ensure similar temperature distributions

### 3. Train Models

```bash
python train_fused.py \
    --splits_root_dir ./dataset_splits \
    --model_type MPNN \
    --epochs 400 \
    --batch_size 16 \
    --lr 0.0005 \
    --gnn_layers 3 \
    --gnn_hidden_dim 128
```

**Available model types**: `MPNN`, `GCN`, `GIN`, `GAT`, `GATv2`, `PAN`, `Transformer`

### 4. Train Specific Folds

```bash
python train_fused.py \
    --splits_root_dir ./dataset_splits \
    --model_type MPNN \
    --folds 1 2 3
```

## Model Architecture

```
Input: Molecular Graph (atoms + bonds) + Proportions Vector
    │
    ├── Atom Embedding Layer
    │       │
    │       ▼
    ├── GNN Layers (with residual connections)
    │   ├── BatchNorm → Activation → GNN Conv
    │   └── Residual Connection
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

## Output Structure

After training, the following outputs are generated:

```
training_output/
└── {MODEL}_lr{LR}_bs{BS}_layers{L}_{TIMESTAMP}/
    ├── logs/
    │   ├── training_config.json    # Training hyperparameters
    │   ├── fold_results.csv        # Per-fold metrics
    │   └── final_summary.csv       # Aggregated metrics
    ├── models/
    │   └── fold_{N}/
    │       ├── model_fold{N}.pth   # Model weights
    │       ├── scaler_fold{N}.joblib
    │       └── feature_stats_fold{N}.json
    └── plot_data/
        └── fold_{N}/
            ├── loss_curve.png
            ├── prediction_scatter.png
            └── epoch_metrics.npz
```

## Evaluation Metrics

- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **R²**: Coefficient of Determination
- **Pearson r**: Pearson Correlation Coefficient
- **Spearman ρ**: Spearman Rank Correlation
- **Kendall τ**: Kendall Tau Correlation

## Citation

If you use this code in your research, please cite:

```bibtex
@article{your_paper,
  title={Your Paper Title},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

## License

MIT License

## Contact

For questions or issues, please open an issue on GitHub.

