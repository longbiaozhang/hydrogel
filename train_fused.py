import torch
from torch import nn, optim
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
import copy
import shutil
import argparse
import joblib
import json
import traceback
try:
    import optuna
except ImportError:
    optuna = None
import ast 
import re 

from hydrogel_gnn_temp_predictor_fused import (
    HydrogelDataset, _get_feature_dims, calculate_feature_stats, save_feature_stats,
    MPNNPredictor, GCNPredictor, GINPredictor, GATPredictor, GATv2Predictor, PANPredictor, TransformerPredictor
)
from metrics import regression_metrics

# --- Plotting functions ---
def plot_loss_curve(train_losses, val_losses, filename, split_num=None):
    plt.figure(figsize=(8, 5)); plt.plot(train_losses, label='Train Loss'); plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss (Scaled)'); plt.title(f'Loss Curve (Split {split_num})')
    plt.legend(); plt.grid(True); plt.tight_layout(); os.makedirs(os.path.dirname(filename), exist_ok=True); plt.savefig(filename); plt.close()

def plot_prediction_vs_true(preds, trues, filename, split_num=None):
    plt.figure(figsize=(6, 6))
    if len(trues) > 0 and len(preds) > 0 and len(trues) == len(preds):
        plt.scatter(trues, preds, alpha=0.6); min_val, max_val = min(np.min(trues), np.min(preds)), max(np.max(trues), np.max(preds))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
    plt.xlabel("True Temperature (Original Scale)"); plt.ylabel("Predicted Temperature (Original Scale)")
    plt.title(f'Prediction vs. True (Validation Split {split_num})'); plt.grid(True); plt.axis('equal'); plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True); plt.savefig(filename); plt.close()

# --- Training and evaluation functions ---
def train(model, loader, optimizer, loss_fn, device, clip_grad_norm):
    model.train(); total_loss = 0; num_samples = 0
    for batch in loader:
        if batch is None: continue
        batch = batch.to(device); optimizer.zero_grad()
        try:
            out = model(batch); target = batch.y.view_as(out); loss = loss_fn(out, target)
            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                if clip_grad_norm > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
                optimizer.step()
                total_loss += loss.item() * batch.num_graphs; num_samples += batch.num_graphs
        except Exception as e: print(f"RuntimeError in train loop: {e}\n{traceback.format_exc()}")
    return total_loss / num_samples if num_samples > 0 else float('inf')

def evaluate(model, loader, loss_fn, device):
    model.eval(); total_loss = 0; num_samples = 0; preds_list, trues_list = [], []
    with torch.no_grad():
        for batch in loader:
            if batch is None: continue
            batch = batch.to(device)
            try:
                out = model(batch); target = batch.y.view_as(out); loss = loss_fn(out, target)
                if not torch.isnan(loss) and not torch.isinf(loss): total_loss += loss.item() * batch.num_graphs; num_samples += batch.num_graphs
                preds_list.append(out.cpu()); trues_list.append(batch.y.cpu())
            except Exception as e: print(f"RuntimeError in eval loop: {e}\n{traceback.format_exc()}")
    final_preds = torch.cat(preds_list) if preds_list else torch.empty(0)
    final_trues = torch.cat(trues_list) if trues_list else torch.empty(0)
    return (total_loss / num_samples) if num_samples > 0 else float('inf'), final_preds, final_trues

# --- train_on_splits  ---
DEFAULT_METRIC_KEYS = ['val_loss_scaled', 'mae', 'mse', 'rmse', 'r2', 'pr', 'sr', 'kr']

def train_on_splits(args, trial=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Starting Cross-Validation for Model: {args.model_type} ---")
    
    if not os.path.exists(args.splits_root_dir): print(f"Error: Split data directory '{args.splits_root_dir}' not found."); return
    split_dirs_all = sorted([os.path.join(args.splits_root_dir, d) for d in os.listdir(args.splits_root_dir) if os.path.isdir(os.path.join(args.splits_root_dir, d)) and d.startswith('split_')])
    if not split_dirs_all: print(f"Error: No 'split_*' subdirectories found in '{args.splits_root_dir}'."); return

    split_dirs = split_dirs_all
    if args.folds:
        print(f"--- FOLD SELECTION: Running only on specified folds: {args.folds} ---")
        selected_dirs = []
        for d in split_dirs_all:
            match = re.search(r'split_(\d+)', os.path.basename(d))
            if match and int(match.group(1)) in args.folds:
                selected_dirs.append(d)
        if not selected_dirs:
            print(f"Error: No directories found matching specified folds {args.folds}.")
            return
        split_dirs = selected_dirs

    all_fold_results = []
    log_dir = os.path.join(args.base_output_dir, "logs"); os.makedirs(log_dir, exist_ok=True)
    models_base_dir = os.path.join(args.base_output_dir, "models"); os.makedirs(models_base_dir, exist_ok=True)
    plot_data_base_dir = os.path.join(args.base_output_dir, "plot_data"); os.makedirs(plot_data_base_dir, exist_ok=True)
    
    with open(os.path.join(log_dir, "training_config.json"), 'w') as f:
        excluded_types = [argparse.Namespace]
        if optuna is not None:
            excluded_types.append(optuna.trial.Trial)
        args_dict = {k: v for k, v in vars(args).items() if not isinstance(v, tuple(excluded_types))}
        json.dump(args_dict, f, indent=4)
    
    
    for fold_idx, split_dir_path in enumerate(split_dirs):
        match = re.search(r'split_(\d+)', os.path.basename(split_dir_path))
        fold_num = int(match.group(1)) if match else fold_idx + 1

        print(f"\n===== [ FOLD {fold_num}/{len(split_dirs_all)} ] =====")
        
        try:
          
            train_df = pd.read_csv(os.path.join(split_dir_path, 'train.csv'))
            val_df = pd.read_csv(os.path.join(split_dir_path, 'validation.csv'))
            
            if 'proportions' in train_df.columns and isinstance(train_df['proportions'].iloc[0], str):
                train_df['proportions'] = train_df['proportions'].apply(ast.literal_eval)
            if 'proportions' in val_df.columns and isinstance(val_df['proportions'].iloc[0], str):
                val_df['proportions'] = val_df['proportions'].apply(ast.literal_eval)

            feature_stats = calculate_feature_stats(train_df)
            scaler = StandardScaler().fit(train_df[['temp_numeric']])
            train_df['y_scaled'] = scaler.transform(train_df[['temp_numeric']])
            val_df['y_scaled'] = scaler.transform(val_df[['temp_numeric']])

            fold_cache_dir = os.path.join(args.root_dir, args.run_name_for_cache, f"fold_{fold_num}_cache")
            if os.path.exists(fold_cache_dir): shutil.rmtree(fold_cache_dir)

            train_dataset = HydrogelDataset(train_df, feature_stats, root=os.path.join(fold_cache_dir, "train"), augment_smiles=True)
            val_dataset = HydrogelDataset(val_df, feature_stats, root=os.path.join(fold_cache_dir, "val"))

            if not train_dataset or not val_dataset: print("Data loading failed, skipping this fold."); continue

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            
            node_dim, edge_dim = _get_feature_dims(feature_stats)
            model_params = {'node_feature_dim': node_dim, 'edge_feature_dim': edge_dim, 'proportion_dim': 5, **vars(args)}
            model_map = {'MPNN': MPNNPredictor, 'GCN': GCNPredictor, 'GIN': GINPredictor, 'GAT': GATPredictor, 'GATv2': GATv2Predictor, 'PAN': PANPredictor, 'Transformer': TransformerPredictor}
            model = model_map[args.model_type](**model_params).to(device)
            
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            loss_fn = nn.HuberLoss() if args.loss_func == 'Huber' else nn.MSELoss()
            
            scheduler = None
            if args.scheduler == 'Cosine':
                if args.epochs <= args.warmup_epochs:
                    print("Warning: Total epochs <= warmup epochs, Cosine scheduler will not be effective.")
                    T_max = 1 
                else:
                    T_max = args.epochs - args.warmup_epochs
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=1e-6)
            elif args.scheduler == 'Step':
                 scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
            elif args.scheduler == 'Plateau':
                 scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)

            best_val_loss, patience_counter = float('inf'), 0
            best_model_state = None
            train_losses, val_losses, val_r2s, val_maes = [], [], [], []

            print(f"--- FOLD {fold_num}: Starting training for {args.epochs} epochs... (Scheduler: {args.scheduler}, Warmup: {args.warmup_epochs} epochs) ---")
            for epoch in range(1, args.epochs + 1):
                if epoch <= args.warmup_epochs and args.warmup_epochs > 0:
                    lr_scale = epoch / args.warmup_epochs
                    for pg in optimizer.param_groups:
                        pg['lr'] = args.lr * lr_scale
                elif scheduler is not None and args.scheduler != 'Plateau':
                    scheduler.step()

                train_loss = train(model, train_loader, optimizer, loss_fn, device, args.clip_grad_norm)
                val_loss, preds_s, trues_s = evaluate(model, val_loader, loss_fn, device)
                
                if scheduler is not None and args.scheduler == 'Plateau' and epoch > args.warmup_epochs:
                    scheduler.step(val_loss)
                
                train_losses.append(train_loss); val_losses.append(val_loss)
                if preds_s.nelement() > 0:
                    preds_o = scaler.inverse_transform(preds_s.numpy().reshape(-1, 1)).flatten()
                    trues_o = val_df['temp_numeric'].values
                    epoch_metrics = regression_metrics(trues_o, preds_o, just_main_metric=False)
                    val_r2s.append(epoch_metrics.get('r2', np.nan))
                    val_maes.append(epoch_metrics.get('mae', np.nan))
                else:
                    val_r2s.append(np.nan); val_maes.append(np.nan)
                
                if epoch % 20 == 0 or epoch == 1 or epoch == args.epochs:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"  Epoch {epoch:03d} | LR: {current_lr:.6f} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val R2: {val_r2s[-1]:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss, patience_counter = val_loss, 0
                    best_model_state = copy.deepcopy(model.state_dict())
                else:
                    patience_counter += 1
                if patience_counter >= args.patience: print(f"  Early stopping at epoch {epoch}."); break
            
            print(f"--- FOLD {fold_num}: Evaluating best model... ---")
            fold_metrics = {k: np.nan for k in DEFAULT_METRIC_KEYS}; fold_metrics['status'] = 'failed'
            if best_model_state:
                model.load_state_dict(best_model_state)
                _, final_preds_s, _ = evaluate(model, val_loader, loss_fn, device)
                if final_preds_s.nelement() > 0:
                    final_preds_o = scaler.inverse_transform(final_preds_s.numpy().reshape(-1, 1)).flatten()
                    final_trues_o = val_df['temp_numeric'].values
                    fold_metrics = regression_metrics(final_trues_o, final_preds_o, just_main_metric=False)
                    fold_metrics['val_loss_scaled'] = best_val_loss
                    fold_metrics['status'] = 'completed'

                    fold_model_dir = os.path.join(models_base_dir, f"fold_{fold_num}"); os.makedirs(fold_model_dir, exist_ok=True)
                    torch.save(best_model_state, os.path.join(fold_model_dir, f"model_fold{fold_num}.pth"))
                    joblib.dump(scaler, os.path.join(fold_model_dir, f"scaler_fold{fold_num}.joblib"))
                    save_feature_stats(feature_stats, os.path.join(fold_model_dir, f"feature_stats_fold{fold_num}.json"))

                    fold_plot_dir = os.path.join(plot_data_base_dir, f"fold_{fold_num:02d}"); os.makedirs(fold_plot_dir, exist_ok=True)
                    plot_loss_curve(train_losses, val_losses, os.path.join(fold_plot_dir, "loss_curve.png"), fold_num)
                    plot_prediction_vs_true(final_preds_o, final_trues_o, os.path.join(fold_plot_dir, "prediction_scatter.png"), fold_num)
                    np.savez(os.path.join(fold_plot_dir, 'epoch_metrics.npz'), train_losses=train_losses, val_losses=val_losses, val_r2s=val_r2s, val_maes=val_maes)
                    np.savez(os.path.join(fold_plot_dir, 'final_predictions.npz'), predictions=final_preds_o, trues=final_trues_o)
            
            all_fold_results.append({'fold': fold_num, **fold_metrics})
            print(f"--- FOLD {fold_num}: Finished. RMSE={fold_metrics.get('rmse', 'N/A'):.4f}, R2={fold_metrics.get('r2', 'N/A'):.4f} ---")
            
        except Exception as e:
            print(f"!!!!!! FOLD {fold_num} FAILED with a critical error: {e} !!!!!!")
            traceback.print_exc()
            all_fold_results.append({'fold': fold_num, 'status': 'crashed', **{k: np.nan for k in DEFAULT_METRIC_KEYS}})
    
    print("\n\n--- All Folds Completed. Generating final summary. ---")
    results_df = pd.DataFrame(all_fold_results)
    results_df.to_csv(os.path.join(log_dir, "fold_results.csv"), index=False, float_format='%.6f')
    
    completed_df = results_df[results_df['status'] == 'completed']
    if not completed_df.empty:
        summary = {}
        for metric in ['r2', 'mae', 'mse', 'rmse', 'r2', 'pr', 'sr', 'kr']:
            summary[f'avg_{metric}'] = completed_df[metric].mean()
            summary[f'std_{metric}'] = completed_df[metric].std()
        pd.DataFrame([summary]).to_csv(os.path.join(log_dir, "final_summary.csv"), index=False, float_format='%.6f')
        print("Final summary of metrics (avg ± std):")
        for k, v in summary.items(): print(f"  {k}: {v:.4f}")
    else:
        print("No folds were completed successfully. No summary generated.")
    
    if trial:
        if not completed_df.empty and 'r2' in completed_df.columns:
            mean_r2 = completed_df['r2'].mean()
            if np.isfinite(mean_r2):
                return mean_r2
        return -1.0  # Return a poor value if failed or no valid R2


# --- Parser definition moved to global scope ---
parser = argparse.ArgumentParser(description="Train GNN models on pre-split datasets.")
# --- Core training parameters ---
parser.add_argument('--splits_root_dir', type=str, default="./dataset_splits", help="Root directory of pre-split datasets.")
parser.add_argument('--model_type', type=str, default='MPNN', choices=['MPNN', 'GCN', 'GIN', 'GAT', 'GATv2', 'PAN', 'Transformer'], help="GNN model type to train.")
parser.add_argument('--epochs', type=int, default=400, help="Maximum number of training epochs.")
parser.add_argument('--patience', type=int, default=100, help="Early stopping patience (epochs).")
parser.add_argument('--batch_size', type=int, default=16, help="Training batch size.")
parser.add_argument('--loss_func', type=str, default='Huber', choices=['MSE', 'Huber'], help="Loss function type.")

# --- Optimizer and scheduler parameters ---
parser.add_argument('--lr', type=float, default=0.0005, help="Base learning rate.")
parser.add_argument('--weight_decay', type=float, default=1e-4, help="Weight decay for AdamW.")
parser.add_argument('--scheduler', type=str, default='Cosine', choices=['Cosine', 'Step', 'Plateau', 'None'], help="Learning rate scheduler type.")
parser.add_argument('--warmup_epochs', type=int, default=50, help="Number of linear warmup epochs.")
parser.add_argument('--clip_grad_norm', type=float, default=1.0, help="Max norm for gradient clipping, 0 to disable.")

# --- Model architecture parameters ---
parser.add_argument('--gnn_layers', type=int, default=3, help="Number of GNN layers.")
parser.add_argument('--gnn_hidden_dim', type=int, default=128, help="GNN hidden dimension.")
parser.add_argument('--mlp_hidden_dim', type=int, default=256, help="Final MLP hidden dimension.")
parser.add_argument('--dropout_rate', type=float, default=0.3, help="Dropout rate.")


parser.add_argument('--output_dir', type=str, default="./training_output", help="Root directory for all outputs.")
parser.add_argument('--root_dir', type=str, default='./dataset_cache', help="Root directory for PyG dataset cache.")
parser.add_argument('--num_workers', type=int, default=0, help="Number of workers for data loading.")
parser.add_argument('--folds', type=int, nargs='+', default=None, help="Specific fold numbers to run (e.g., --folds 7 8 10). If not provided, run all folds.")



if __name__ == '__main__':
    # In __main__ block, only parse arguments and call main function
    args = parser.parse_args()
    
    # Smart directory generation 
    timestamp = int(time.time())
    run_name_base = f"{args.model_type}_lr{args.lr}_bs{args.batch_size}_layers{args.gnn_layers}"
    if args.folds:
        run_name_base += f"_folds{''.join(map(str, args.folds))}"
        
    run_name_final = f"{run_name_base}_{timestamp}"
    args.run_name_for_cache = run_name_final
    args.base_output_dir = os.path.join(args.output_dir, run_name_final)
    os.makedirs(args.base_output_dir, exist_ok=True)
    
    print(f"--- Standalone run start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))} ---")
    print(f"Full run parameters: {vars(args)}")
    print(f"All outputs will be saved to: {args.base_output_dir}")
    
    train_on_splits(args)
    
    print(f"\nTraining pipeline completed!\nAll results saved in: {args.base_output_dir}")