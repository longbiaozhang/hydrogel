# create_dataset_splits.py
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import wasserstein_distance
import os
import argparse
import numpy as np
import re
from tqdm import tqdm

def load_and_preprocess_for_splitting(filepath, is_excel=True):
    """
    A robust loading function for reading and preprocessing data, specifically for splitting.
    It handles column names and temperature parsing to ensure a clean DataFrame.
    """
    print(f"Loading data from '{filepath}'...")
    df = None
    try:
        if is_excel:
            df = pd.read_excel(filepath, header=0)
        else:
            df = pd.read_csv(filepath, header=0)
        print(f"File '{filepath}' read successfully, containing {len(df)} rows.")
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

    if df is None or df.empty:
        print("File is empty or parsing failed after loading.")
        return None

    # --- Compatibility handling: ensure key columns exist ---
    if 'SMILES' not in df.columns or 'temp_raw' not in df.columns:
        print("Warning: Missing 'SMILES' or 'temp_raw' columns. Attempting heuristic renaming based on position...")
        num_cols = df.shape[1]
        expected_base_cols = ["SMILES", "SMILES_prop", "ligand1", "ligand1_prop", "ligand2", "ligand2_prop", 
                              "ligand3", "ligand3_prop", "ligand4", "ligand4_prop", "temp_raw", "paper"]
        cols_to_assign = {df.columns[i]: expected_base_cols[i] for i in range(min(num_cols, len(expected_base_cols)))}
        df.rename(columns=cols_to_assign, inplace=True)

    if 'SMILES' not in df.columns or 'temp_raw' not in df.columns:
        print(f"Error: Still missing 'SMILES' or 'temp_raw' columns after renaming. Current columns: {df.columns.tolist()}")
        return None
    
    # --- Parse temperature column ---
    def parse_temp(temp_str):
        if pd.isna(temp_str): return np.nan
        if isinstance(temp_str, (int, float)): return float(temp_str)
        if isinstance(temp_str, str):
            ts = temp_str.strip().replace(' ','')
            range_match = re.match(r"^(-?\d+\.?\d*)\s*-\s*(-?\d+\.?\d*)$", ts)
            if range_match: return (float(range_match.group(1)) + float(range_match.group(2))) / 2.0
            single_num_match = re.match(r"^[><≥≤≈~]?\s*(-?\d+\.?\d*(?:[eE][+-]?\d+)?)", ts)
            if single_num_match: return float(single_num_match.group(1))
        return np.nan
        
    print("Parsing 'temp_raw' column to create 'temp_numeric'...")
    df['temp_numeric'] = df['temp_raw'].apply(parse_temp)
    
    # --- Clean data ---
    df.dropna(subset=['temp_numeric', 'SMILES'], inplace=True)
    df = df[df['SMILES'].astype(str).str.strip() != '']
    df = df.reset_index(drop=True)
    
    if df.empty:
        print("Error: Data is empty after cleaning.")
        return None

    # --- Prepare proportions column for subsequent training scripts ---
    max_ligands = 4
    prop_vec_cols = ['SMILES_prop']
    for i in range(1, max_ligands + 1):
        prop_vec_cols.append(f"ligand{i}_prop")
    
    for col in prop_vec_cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    
    df['proportions'] = df[prop_vec_cols].values.tolist()
        
    print(f"Data loading and preprocessing completed, {len(df)} valid records.")
    return df

def evaluate_split_quality(original_dist, train_dist, val_dist, test_dist):
    ws_train = wasserstein_distance(original_dist, train_dist)
    ws_val = wasserstein_distance(original_dist, val_dist)
    ws_test = wasserstein_distance(original_dist, test_dist)
    return ws_train + ws_val + ws_test

def create_and_save_splits(args):
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Sum of train, val, test ratios must be 1, currently: {total_ratio}")

    df_full = load_and_preprocess_for_splitting(args.data_path, args.is_excel)
    if df_full is None: return
    original_dist = df_full['temp_numeric'].values
    
    print(f"\n--- Finding and saving the best {args.n_splits_to_save} splits from {args.n_candidates} candidates ---")
    
    all_splits_info = []
    seeds = range(args.random_seed_start, args.random_seed_start + args.n_candidates)

    for seed in tqdm(seeds, desc="Generating and evaluating candidate splits"):
        train_val_df, test_df = train_test_split(
            df_full, test_size=args.test_ratio, random_state=seed)
        
        val_ratio_in_train_val = args.val_ratio / (args.train_ratio + args.val_ratio)
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_ratio_in_train_val, random_state=seed)
        
        quality_score = evaluate_split_quality(
            original_dist,
            train_df['temp_numeric'].values,
            val_df['temp_numeric'].values,
            test_df['temp_numeric'].values
        )
        
        all_splits_info.append({
            'seed': seed, 'quality_score': quality_score, 'train_df': train_df,
            'val_df': val_df, 'test_df': test_df
        })

    ranked_splits = sorted(all_splits_info, key=lambda x: x['quality_score'])
    best_splits = ranked_splits[:args.n_splits_to_save]

    print(f"\n--- Selected {args.n_splits_to_save} best splits, saving files... ---")
    
    os.makedirs(args.output_dir, exist_ok=True)
    summary_data = []
    for i, split_data in enumerate(best_splits):
        split_dir_name = f"split_{i+1:02d}"
        split_path = os.path.join(args.output_dir, split_dir_name)
        os.makedirs(split_path, exist_ok=True)
        
        split_data['train_df'].to_csv(os.path.join(split_path, 'train.csv'), index=False)
        split_data['val_df'].to_csv(os.path.join(split_path, 'validation.csv'), index=False)
        split_data['test_df'].to_csv(os.path.join(split_path, 'test.csv'), index=False)
        
        print(f"  Saved split {i+1} to directory: '{split_path}' (train: {len(split_data['train_df'])}, val: {len(split_data['val_df'])}, test: {len(split_data['test_df'])})")
        summary_data.append({
            'split_id': i + 1, 'directory': split_dir_name, 'seed': split_data['seed'],
            'quality_score': split_data['quality_score'], 'train_size': len(split_data['train_df']),
            'val_size': len(split_data['val_df']), 'test_size': len(split_data['test_df']),
        })
        
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(args.output_dir, 'splits_summary.csv'), index=False)
    print(f"\nSummary of all splits saved to: '{os.path.join(args.output_dir, 'splits_summary.csv')}'")
    print("\n--- Operation completed ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate and save multiple high-quality Train/Val/Test dataset splits.")
    parser.add_argument('--data_path', type=str, default="SMILES.xlsx", help="Path to source file containing complete data.")
    parser.add_argument('--is_excel', action='store_true', default=True, help="Whether source file is in Excel format.")
    parser.add_argument('--output_dir', type=str, default="./dataset_splits", help="Root directory to save all split files.")
    parser.add_argument('--train_ratio', type=float, default=0.7, help="Training set ratio.")
    parser.add_argument('--val_ratio', type=float, default=0.15, help="Validation set ratio.")
    parser.add_argument('--test_ratio', type=float, default=0.15, help="Test set ratio.")
    parser.add_argument('--n_splits_to_save', type=int, default=10, help="Number of best splits to save.")
    parser.add_argument('--n_candidates', type=int, default=50, help="Number of candidate splits for selection.")
    parser.add_argument('--random_seed_start', type=int, default=42, help="Starting random seed for generating candidate splits.")
    args = parser.parse_args()
    create_and_save_splits(args)