# Data Template Guide

## Overview

This project requires a dataset in Excel (.xlsx) or CSV format. The original dataset used in our research is **not publicly available**. Users need to prepare their own data following the format described below.

## Required Data Format

### Excel/CSV Columns

Your data file should contain the following columns:

| Column Name | Type | Description | Example |
|-------------|------|-------------|---------|
| `SMILES` | String | Main polymer SMILES string | `CC(C)OC(=O)C(C)C` |
| `SMILES_prop` | Float | Proportion of main polymer (0-1) | `0.7` |
| `ligand1` | String | First ligand SMILES (optional) | `CCO` |
| `ligand1_prop` | Float | First ligand proportion | `0.1` |
| `ligand2` | String | Second ligand SMILES (optional) | `CC(C)O` |
| `ligand2_prop` | Float | Second ligand proportion | `0.1` |
| `ligand3` | String | Third ligand SMILES (optional) | `CCCO` |
| `ligand3_prop` | Float | Third ligand proportion | `0.05` |
| `ligand4` | String | Fourth ligand SMILES (optional) | `CCC(C)O` |
| `ligand4_prop` | Float | Fourth ligand proportion | `0.05` |
| `temp_raw` | String/Float | Target temperature in °C | `32.5` or `30-35` |
| `paper` | String | Reference (optional) | `Smith et al. 2024` |

### Important Notes

1. **SMILES Format**: All SMILES strings must be valid RDKit-readable SMILES
2. **Proportions**: All proportion values should sum to 1.0 for each sample
3. **Temperature**: Can be:
   - Single value: `32.5`
   - Range: `30-35` (will be averaged)
   - With operators: `>30`, `≥32`, `~35`
4. **Missing Ligands**: If a sample has fewer than 4 ligands, leave those columns empty or set proportion to 0

## Example Data

### Minimal Example (CSV format)

```csv
SMILES,SMILES_prop,ligand1,ligand1_prop,ligand2,ligand2_prop,ligand3,ligand3_prop,ligand4,ligand4_prop,temp_raw,paper
CC(C)OC(=O)C(C)C,1.0,,0.0,,0.0,,0.0,,0.0,32.5,Example 2024
CCO,0.8,CC(C)O,0.2,,0.0,,0.0,,0.0,28.3,Example 2024
CCCO,0.7,CCO,0.2,CC(C)O,0.1,,0.0,,0.0,35.0,Example 2024
```

### Full Example (Excel format)

| SMILES | SMILES_prop | ligand1 | ligand1_prop | ligand2 | ligand2_prop | ligand3 | ligand3_prop | ligand4 | ligand4_prop | temp_raw | paper |
|--------|-------------|---------|--------------|---------|--------------|---------|--------------|---------|--------------|----------|-------|
| CC(C)OC(=O)C(C)C | 0.70 | CCO | 0.15 | CC(C)O | 0.10 | CCCO | 0.03 | CCC(C)O | 0.02 | 32.5 | Smith 2024 |
| CCCCOC(=O)C | 0.85 | CCO | 0.10 | CC(C)O | 0.05 | | 0.00 | | 0.00 | 28-30 | Jones 2024 |
| CC(C)C | 1.00 | | 0.00 | | 0.00 | | 0.00 | | 0.00 | >35 | Lee 2024 |

## Data Requirements

### Minimum Dataset Size
- **Recommended**: At least 100-200 samples for meaningful training
- **Minimum**: 50 samples (may result in poor performance)

### Data Quality
- All SMILES must be valid and parseable by RDKit
- Temperature values must be numeric or parseable ranges
- No duplicate samples (same SMILES + ligand combinations)
- Balanced distribution of temperature values recommended

## Creating Your Dataset

### Step 1: Prepare Data
Create an Excel file with the required columns using the template above.

### Step 2: Validate SMILES
```python
from rdkit import Chem

def validate_smiles(smiles_string):
    mol = Chem.MolFromSmiles(smiles_string)
    return mol is not None

# Test your SMILES
test_smiles = "CC(C)OC(=O)C(C)C"
print(f"Valid: {validate_smiles(test_smiles)}")
```

### Step 3: Check Proportions
Ensure all proportions sum to 1.0 for each row:
```python
import pandas as pd

df = pd.read_excel("your_data.xlsx")
prop_cols = ['SMILES_prop', 'ligand1_prop', 'ligand2_prop', 'ligand3_prop', 'ligand4_prop']
prop_sums = df[prop_cols].sum(axis=1)
print(f"All proportions sum to 1.0: {all(abs(prop_sums - 1.0) < 0.01)}")
```

### Step 4: Use with the Code
```bash
python create_dataset_splits.py --data_path your_data.xlsx
```

## Sample Data Generator (Optional)

If you want to test the code with synthetic data:

```python
import pandas as pd
import numpy as np

# Generate synthetic data
n_samples = 100
data = {
    'SMILES': ['CCO'] * n_samples,  # Replace with real SMILES
    'SMILES_prop': np.random.uniform(0.7, 1.0, n_samples),
    'ligand1': ['CC(C)O'] * n_samples,
    'ligand1_prop': np.random.uniform(0, 0.3, n_samples),
    'ligand2': [''] * n_samples,
    'ligand2_prop': [0.0] * n_samples,
    'ligand3': [''] * n_samples,
    'ligand3_prop': [0.0] * n_samples,
    'ligand4': [''] * n_samples,
    'ligand4_prop': [0.0] * n_samples,
    'temp_raw': np.random.uniform(25, 40, n_samples),
    'paper': ['Synthetic'] * n_samples
}

# Normalize proportions
df = pd.DataFrame(data)
prop_cols = ['SMILES_prop', 'ligand1_prop', 'ligand2_prop', 'ligand3_prop', 'ligand4_prop']
row_sums = df[prop_cols].sum(axis=1)
for col in prop_cols:
    df[col] = df[col] / row_sums

df.to_excel('synthetic_data.xlsx', index=False)
```

## Troubleshooting

### Common Issues

1. **"Missing 'SMILES' or 'temp_raw' columns"**
   - Check column names match exactly (case-sensitive)
   - Ensure no extra spaces in column names

2. **"Invalid SMILES"**
   - Validate all SMILES using RDKit
   - Remove or fix invalid SMILES strings

3. **"Proportions don't sum to 1"**
   - Check all proportion columns
   - Use the validation script above

4. **"Data is empty after cleaning"**
   - Check for NaN values in required columns
   - Ensure temperature values are parseable

## Contact

For questions about data format or preparation, please open an issue on GitHub.
