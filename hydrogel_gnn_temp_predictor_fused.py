import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import (MessagePassing, Set2Set, GCNConv, GINConv,
                                BatchNorm as GraphBatchNorm, GATConv, GATv2Conv, PANConv, TransformerConv)
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors 
from tqdm.auto import tqdm
import os
import numpy as np
import re
from collections import defaultdict
import json
import traceback

def load_and_preprocess_data(file_path, is_excel=True):
    """
    Load data from Excel or CSV file into a pandas DataFrame.
    """
    if not os.path.exists(file_path):
        print(f"Error: Data file '{file_path}' not found.")
        return None
    try:
        print(f"Loading SMILES data from {file_path}...")
        if is_excel:
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
        print("Data loaded successfully.")
        return df
    except Exception as e:
        print(f"Error loading file '{file_path}': {e}")
        traceback.print_exc()
        return None
    
# --- Feature extraction functions ---
def calculate_feature_stats(dataframe, smiles_col='SMILES'):
    print(f"Calculating feature statistics from {len(dataframe)} records...")
    feature_values = defaultdict(list)
    if smiles_col not in dataframe.columns:
        print(f"Error: Missing '{smiles_col}' column in DataFrame.")
        return None
    for smiles in tqdm(dataframe[smiles_col], desc="Computing feature stats"):
        if not smiles or pd.isna(smiles): continue
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: continue
            for atom in mol.GetAtoms():
                feature_values['degree'].append(float(atom.GetDegree()))
                feature_values['charge'].append(float(atom.GetFormalCharge()))
                feature_values['num_hs'].append(float(atom.GetTotalNumHs(includeNeighbors=True)))
                feature_values['total_valence'].append(float(atom.GetTotalValence()))
                feature_values['num_radical_electrons'].append(float(atom.GetNumRadicalElectrons()))
                feature_values['hybridization_numeric'].append(float(atom.GetHybridization().real))
        except Exception as e:
            print(f"Warning: Error processing SMILES '{smiles}': {e}")
    stats = {}
    numerical_features_to_normalize = ['degree', 'charge', 'num_hs', 'total_valence', 'num_radical_electrons', 'hybridization_numeric']
    for name in numerical_features_to_normalize:
        values = feature_values.get(name, [])
        if not values:
            mean, std = 0.0, 1.0
            print(f"Warning: No values collected for feature '{name}', using default mean=0, std=1.")
        else:
            tensor_values = torch.tensor(values, dtype=torch.float)
            mean = torch.mean(tensor_values).item()
            std = torch.std(tensor_values).item()
            if std < 1e-8: std = 1.0
        stats[name] = {'mean': mean, 'std': std}
    return stats

def save_feature_stats(stats, filepath="feature_stats.json"):
    try:
        dir_name = os.path.dirname(filepath)
        if dir_name: os.makedirs(dir_name, exist_ok=True)
        with open(filepath, 'w') as f: json.dump(stats, f, indent=4)
    except Exception as e: print(f"Error saving feature stats to {filepath}: {e}")

def get_atom_features(atom, feature_stats=None):
    atom_type_idx = float(atom.GetAtomicNum())
    raw_degree = float(atom.GetDegree())
    raw_charge = float(atom.GetFormalCharge())
    raw_num_hs = float(atom.GetTotalNumHs(includeNeighbors=True))
    raw_total_valence = float(atom.GetTotalValence())
    raw_num_radical_electrons = float(atom.GetNumRadicalElectrons())
    raw_hybridization_numeric = float(atom.GetHybridization().real)
    degree, charge, num_hs = raw_degree, raw_charge, raw_num_hs
    total_valence, num_radical_electrons, hybridization_numeric = raw_total_valence, raw_num_radical_electrons, raw_hybridization_numeric
    if feature_stats:
        if 'degree' in feature_stats and feature_stats['degree']['std'] > 1e-8:
             degree = (raw_degree - feature_stats['degree']['mean']) / feature_stats['degree']['std']
        if 'charge' in feature_stats and feature_stats['charge']['std'] > 1e-8:
             charge = (raw_charge - feature_stats['charge']['mean']) / feature_stats['charge']['std']
        if 'num_hs' in feature_stats and feature_stats['num_hs']['std'] > 1e-8:
             num_hs = (raw_num_hs - feature_stats['num_hs']['mean']) / feature_stats['num_hs']['std']
        if 'total_valence' in feature_stats and feature_stats['total_valence']['std'] > 1e-8:
            total_valence = (raw_total_valence - feature_stats['total_valence']['mean']) / feature_stats['total_valence']['std']
        if 'num_radical_electrons' in feature_stats and feature_stats['num_radical_electrons']['std'] > 1e-8:
            num_radical_electrons = (raw_num_radical_electrons - feature_stats['num_radical_electrons']['mean']) / feature_stats['num_radical_electrons']['std']
        if 'hybridization_numeric' in feature_stats and feature_stats['hybridization_numeric']['std'] > 1e-8:
            hybridization_numeric = (raw_hybridization_numeric - feature_stats['hybridization_numeric']['mean']) / feature_stats['hybridization_numeric']['std']
    features = [
        atom_type_idx, degree, charge, num_hs,
        float(atom.GetHybridization() == Chem.rdchem.HybridizationType.SP),
        float(atom.GetHybridization() == Chem.rdchem.HybridizationType.SP2),
        float(atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3),
        float(atom.GetIsAromatic()), float(atom.IsInRing()),
        total_valence, num_radical_electrons, hybridization_numeric
    ]
    return torch.tensor(features, dtype=torch.float)

def get_bond_features(bond):
    bt = bond.GetBondType()
    return torch.tensor([
        float(bt == Chem.rdchem.BondType.SINGLE), float(bt == Chem.rdchem.BondType.DOUBLE),
        float(bt == Chem.rdchem.BondType.TRIPLE), float(bt == Chem.rdchem.BondType.AROMATIC),
        float(bond.GetIsConjugated()), float(bond.IsInRing())], dtype=torch.float)

def _get_feature_dims(feature_stats_for_calc=None):
    dummy_mol = Chem.MolFromSmiles('CCO')
    if dummy_mol is None: raise ValueError("Cannot create dummy molecule 'CCO' for feature dimension calculation.")
    atom_dim = len(get_atom_features(dummy_mol.GetAtomWithIdx(0), feature_stats=feature_stats_for_calc))
    bond_dim = len(get_bond_features(dummy_mol.GetBondWithIdx(0)))
    return atom_dim, bond_dim

def smiles_to_graph(smiles_string, feature_stats=None, augment=False):
    atom_feature_dim, bond_feature_dim = _get_feature_dims(feature_stats_for_calc=feature_stats)
    def create_dummy_graph_data():
        return Data(x=torch.zeros((1, atom_feature_dim), dtype=torch.float), edge_index=torch.empty((2, 0), dtype=torch.long), edge_attr=torch.empty((0, bond_feature_dim), dtype=torch.float), is_dummy=True)
    if not smiles_string or pd.isna(smiles_string) or not smiles_string.strip(): return create_dummy_graph_data()
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None: return create_dummy_graph_data()
        if augment:
            try:
                augmented_smiles = AllChem.MolToSmiles(mol, canonical=False, isomericSmiles=True, doRandom=True)
                augmented_mol = Chem.MolFromSmiles(augmented_smiles)
                if augmented_mol is not None: mol = augmented_mol
            except Exception: pass 
    except Exception: return create_dummy_graph_data()
    atom_features_list = [get_atom_features(atom, feature_stats) for atom in mol.GetAtoms()]
    if not atom_features_list: return create_dummy_graph_data()
    x = torch.stack(atom_features_list)
    edge_indices_list, edge_features_list = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_indices_list.extend([(i, j), (j, i)])
        bond_feat = get_bond_features(bond)
        edge_features_list.extend([bond_feat, bond_feat])
    edge_index = torch.tensor(edge_indices_list, dtype=torch.long).t().contiguous() if edge_indices_list else torch.empty((2,0), dtype=torch.long)
    edge_attr = torch.stack(edge_features_list) if edge_features_list else torch.empty((0, bond_feature_dim), dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, is_dummy=False)



def get_molecular_descriptors(smiles):
    """
    Calculate molecular-level chemical descriptors.
    Returns a list of 6 descriptors, or all zeros if SMILES is invalid.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [0.0] * 6
        return [
            Descriptors.MolWt(mol),           # Molecular weight
            Descriptors.MolLogP(mol),         # Lipophilicity (LogP)
            Descriptors.TPSA(mol),            # Topological polar surface area
            Descriptors.NumHDonors(mol),      # Number of H-bond donors
            Descriptors.NumHAcceptors(mol),   # Number of H-bond acceptors
            Descriptors.NumRotatableBonds(mol) # Number of rotatable bonds
        ]
    except Exception:
        return [0.0] * 6


class HydrogelDataset(InMemoryDataset):
    # Number of chemical descriptors (for external dimension retrieval)
    NUM_MOL_DESCRIPTORS = 6
    
    def __init__(self, dataframe, feature_stats, root='.', transform=None, pre_transform=None, augment_smiles=False, use_mol_descriptors=False):
        self.dataframe = dataframe.reset_index(drop=True)
        self.feature_stats = feature_stats
        self.augment_smiles = augment_smiles
        self.use_mol_descriptors = use_mol_descriptors
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self): return []
    @property
    def processed_file_names(self): return ['hydrogel_processed_data.pt']
    def process(self):
        data_list = []
        # Collect all descriptors for normalization
        all_descriptors = []
        if self.use_mol_descriptors:
            for _, row in self.dataframe.iterrows():
                desc = get_molecular_descriptors(row['SMILES'])
                all_descriptors.append(desc)
            all_descriptors = np.array(all_descriptors)
            self.desc_mean = np.mean(all_descriptors, axis=0)
            self.desc_std = np.std(all_descriptors, axis=0)
            self.desc_std[self.desc_std < 1e-8] = 1.0  # Avoid division by zero
        
        for idx, row in tqdm(self.dataframe.iterrows(), total=len(self.dataframe), desc=f"Processing SMILES for {os.path.basename(self.root)} (Augment: {self.augment_smiles})"):
            if 'y_scaled' not in row or pd.isna(row['y_scaled']): continue
            main_graph = smiles_to_graph(row['SMILES'], self.feature_stats, augment=self.augment_smiles)
            if main_graph.is_dummy: continue
            ligand_merged_smiles = row.get('Ligand_merged', None)
            ligand_graph = smiles_to_graph(ligand_merged_smiles, self.feature_stats, augment=False)
            num_main_nodes = main_graph.x.shape[0]
            if ligand_graph.is_dummy:
                combined_x, combined_edge_index, combined_edge_attr = main_graph.x, main_graph.edge_index, main_graph.edge_attr
            else:
                ligand_edge_index_shifted = ligand_graph.edge_index + num_main_nodes
                combined_x = torch.cat([main_graph.x, ligand_graph.x], dim=0)
                combined_edge_index = torch.cat([main_graph.edge_index, ligand_edge_index_shifted], dim=1)
                combined_edge_attr = torch.cat([main_graph.edge_attr, ligand_graph.edge_attr], dim=0)
            
            # Original proportions (5-dim)
            proportions_list = row.get('proportions', [0.0] * 5)
            
            # Add chemical descriptors (6-dim)
            if self.use_mol_descriptors:
                mol_desc = get_molecular_descriptors(row['SMILES'])
                # Normalize descriptors
                mol_desc_normalized = (np.array(mol_desc) - self.desc_mean) / self.desc_std
                # Concatenate: proportions (5) + mol_descriptors (6) = 11-dim
                global_features = proportions_list + mol_desc_normalized.tolist()
            else:
                global_features = proportions_list
            
            data_obj = Data(x=combined_x, edge_index=combined_edge_index, edge_attr=combined_edge_attr, y=torch.tensor([row['y_scaled']], dtype=torch.float), proportions=torch.tensor(global_features, dtype=torch.float).unsqueeze(0))
            data_list.append(data_obj)
        if not data_list:
            data, slices = self.collate([])
        else:
            data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

# --- GNN Models ---
class BaseGNNPredictor(nn.Module):
    def __init__(self, proportion_dim=5, gnn_hidden_dim=128, set2set_processing_steps=6, set2set_layers=2, mlp_hidden_dim=256, dropout_rate=0.3, **kwargs):
        super().__init__()
        self.set2set = Set2Set(gnn_hidden_dim, processing_steps=set2set_processing_steps, num_layers=set2set_layers)
        set2set_output_dim = 2 * gnn_hidden_dim
        proportion_mlp_output_dim = gnn_hidden_dim // 2
        self.proportion_mlp = nn.Sequential(nn.Linear(proportion_dim, gnn_hidden_dim), nn.BatchNorm1d(gnn_hidden_dim), nn.ReLU(), nn.Linear(gnn_hidden_dim, proportion_mlp_output_dim))
        combined_feature_dim = set2set_output_dim + proportion_mlp_output_dim
        self.final_mlp = nn.Sequential(nn.Linear(combined_feature_dim, mlp_hidden_dim), nn.BatchNorm1d(mlp_hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2), nn.BatchNorm1d(mlp_hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(mlp_hidden_dim // 2, 1))
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x_out = self.gnn_forward(x, edge_index, edge_attr)
        graph_embedding = self.set2set(x_out, batch)
        proportions = data.proportions.squeeze(1) if data.proportions.dim() == 3 and data.proportions.shape[1] == 1 else data.proportions
        proportion_embedding = self.proportion_mlp(proportions)
        combined_embedding = torch.cat([graph_embedding, proportion_embedding], dim=1)
        return self.final_mlp(combined_embedding)
    def gnn_forward(self, x, edge_index, edge_attr):
        raise NotImplementedError("Subclasses must implement GNN forward pass.")

class MPNNLayer(MessagePassing):
    def __init__(self, node_channels, edge_channels, output_channels):
        super().__init__(aggr='add')
        self.node_channels = node_channels
        self.output_channels = output_channels
        self.edge_feature_dim = max(edge_channels, 1)
        self.edge_mlp = nn.Sequential(nn.Linear(self.edge_feature_dim, node_channels * output_channels), nn.ReLU())
        self.node_update_mlp = nn.Sequential(nn.Linear(node_channels + output_channels, output_channels), nn.ReLU())
    def forward(self, x, edge_index, edge_attr):
        if edge_attr is None or (edge_attr.nelement() > 0 and edge_attr.shape[1] == 0):
            if edge_index.shape[1] > 0:
                edge_attr = torch.ones(edge_index.shape[1], self.edge_feature_dim, device=x.device, dtype=x.dtype)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    def message(self, x_j, edge_attr):
        edge_mlp_out = self.edge_mlp(edge_attr)
        weights = edge_mlp_out.view(-1, self.node_channels, self.output_channels)
        return torch.matmul(x_j.unsqueeze(1), weights).squeeze(1)
    def update(self, aggr_out, x):
        return self.node_update_mlp(torch.cat([x, aggr_out], dim=1))

# Shared GNN forward logic with pre-activation residual structure ---
def embedding_gnn_forward(self, x, edge_index, edge_attr):
    # 1. Initial feature encoding
    atom_indices = x[:, 0].long()
    other_features = x[:, 1:]
    atom_embeddings = self.atom_embedding_layer(atom_indices)
    combined_features = torch.cat([atom_embeddings, other_features], dim=-1)
    x = self.node_encoder(combined_features)

    # 2. Loop through GNN layers
    for i, layer in enumerate(self.gnn_conv_layers):
        identity = x # Save original features for residual connection

        # --- Pre-activation module (BN -> Activation) ---
        # a. Batch normalization
        processed_x = self.gnn_batch_norms[i](x)
        # b. Activation function
        #    - GAT/GATv2 typically use ELU
        #    - Other models typically use ReLU
        if isinstance(layer, (GATConv, GATv2Conv)):
            processed_x = F.elu(processed_x)
        else:
            processed_x = F.relu(processed_x)
        
        # --- GNN convolution module ---
        if isinstance(layer, (MPNNLayer, TransformerConv, GATConv, GATv2Conv)):
            current_edge_attr = edge_attr
            if hasattr(self, 'effective_edge_dim') and self.effective_edge_dim is not None:
                if edge_attr is None or (edge_attr.nelement() > 0 and edge_attr.shape[1] == 0):
                    current_edge_attr = torch.ones(edge_index.shape[1], self.effective_edge_dim, device=x.device)
            # Pass pre-activated features to GNN layer
            processed_x = layer(processed_x, edge_index, edge_attr=current_edge_attr)

        elif isinstance(layer, (GCNConv, GINConv, PANConv)):
            if isinstance(layer, PANConv):
                processed_x, _ = layer(processed_x, edge_index)
            else:
                # Pass pre-activated features to GNN layer
                processed_x = layer(processed_x, edge_index)
        else:
            raise NotImplementedError(f"GNN layer type {type(layer).__name__} is not handled in gnn_forward.")
        
        # --- Residual connection ---
        x = identity + processed_x
        
    # Final normalization after all GNN layers to stabilize features for Set2Set layer
    x = self.gnn_batch_norms[-1](x)
    
    return x



# --- Shared GNN model initialization logic ---
def embedding_gnn_init(self, node_feature_dim, gnn_hidden_dim, gnn_layers_count):
    self.atom_embedding_dim = 32
    self.atom_embedding_layer = nn.Embedding(num_embeddings=119, embedding_dim=self.atom_embedding_dim)
    num_other_features = node_feature_dim - 1
    encoder_input_dim = self.atom_embedding_dim + num_other_features
    self.node_encoder = nn.Linear(encoder_input_dim, gnn_hidden_dim)
    
    self.gnn_conv_layers = nn.ModuleList()
    self.gnn_batch_norms = nn.ModuleList()
    # Create one BN layer for each GNN layer, plus one extra for final output
    for _ in range(gnn_layers_count + 1):
        self.gnn_batch_norms.append(GraphBatchNorm(gnn_hidden_dim))



# --- All GNN model classes now use the new, more stable structure ---
class MPNNPredictor(BaseGNNPredictor):
    def __init__(self, node_feature_dim, edge_feature_dim, **kwargs):
        super().__init__(**kwargs)
        gnn_layers_count = kwargs.get('gnn_layers', 3)
        gnn_hidden_dim = kwargs.get('gnn_hidden_dim', 128)
        embedding_gnn_init(self, node_feature_dim, gnn_hidden_dim, gnn_layers_count)
        for _ in range(gnn_layers_count):
            self.gnn_conv_layers.append(MPNNLayer(gnn_hidden_dim, edge_feature_dim, gnn_hidden_dim))
    gnn_forward = embedding_gnn_forward

class GCNPredictor(BaseGNNPredictor):
    def __init__(self, node_feature_dim, edge_feature_dim, **kwargs):
        super().__init__(**kwargs)
        gnn_layers_count = kwargs.get('gnn_layers', 3)
        gnn_hidden_dim = kwargs.get('gnn_hidden_dim', 128)
        embedding_gnn_init(self, node_feature_dim, gnn_hidden_dim, gnn_layers_count)
        for _ in range(gnn_layers_count):
            self.gnn_conv_layers.append(GCNConv(gnn_hidden_dim, gnn_hidden_dim))
    gnn_forward = embedding_gnn_forward

class GINPredictor(BaseGNNPredictor):
    def __init__(self, node_feature_dim, edge_feature_dim, **kwargs):
        super().__init__(**kwargs)
        gnn_layers_count = kwargs.get('gnn_layers', 3)
        gnn_hidden_dim = kwargs.get('gnn_hidden_dim', 128)
        embedding_gnn_init(self, node_feature_dim, gnn_hidden_dim, gnn_layers_count)
        for _ in range(gnn_layers_count):
            mlp = nn.Sequential(nn.Linear(gnn_hidden_dim, gnn_hidden_dim*2), nn.ReLU(), nn.Linear(gnn_hidden_dim*2, gnn_hidden_dim))
            self.gnn_conv_layers.append(GINConv(mlp, train_eps=True))
    gnn_forward = embedding_gnn_forward

class GATPredictor(BaseGNNPredictor):
    def __init__(self, node_feature_dim, edge_feature_dim, **kwargs):
        super().__init__(**kwargs)
        gnn_layers_count = kwargs.get('gnn_layers', 3)
        gnn_hidden_dim = kwargs.get('gnn_hidden_dim', 128)
        dropout_rate = kwargs.get('dropout_rate', 0.3)
        heads = 4
        if gnn_hidden_dim % heads != 0: raise ValueError(f"gnn_hidden_dim ({gnn_hidden_dim}) must be divisible by heads ({heads})")
        embedding_gnn_init(self, node_feature_dim, gnn_hidden_dim, gnn_layers_count)
        self.effective_edge_dim = max(1, edge_feature_dim) if edge_feature_dim > 0 else None
        for _ in range(gnn_layers_count):
            self.gnn_conv_layers.append(GATConv(gnn_hidden_dim, gnn_hidden_dim // heads, heads=heads, dropout=dropout_rate, edge_dim=self.effective_edge_dim))
    gnn_forward = embedding_gnn_forward

class GATv2Predictor(BaseGNNPredictor):
    def __init__(self, node_feature_dim, edge_feature_dim, **kwargs):
        super().__init__(**kwargs)
        gnn_layers_count = kwargs.get('gnn_layers', 3)
        gnn_hidden_dim = kwargs.get('gnn_hidden_dim', 128)
        dropout_rate = kwargs.get('dropout_rate', 0.3)
        heads = 4
        if gnn_hidden_dim % heads != 0: raise ValueError(f"gnn_hidden_dim ({gnn_hidden_dim}) must be divisible by heads ({heads})")
        embedding_gnn_init(self, node_feature_dim, gnn_hidden_dim, gnn_layers_count)
        self.effective_edge_dim = max(1, edge_feature_dim) if edge_feature_dim > 0 else None
        for _ in range(gnn_layers_count):
            self.gnn_conv_layers.append(GATv2Conv(gnn_hidden_dim, gnn_hidden_dim // heads, heads=heads, dropout=dropout_rate, concat=True, edge_dim=self.effective_edge_dim))
    gnn_forward = embedding_gnn_forward

class PANPredictor(BaseGNNPredictor):
    def __init__(self, node_feature_dim, edge_feature_dim, **kwargs):
        super().__init__(**kwargs)
        gnn_layers_count = kwargs.get('gnn_layers', 3)
        gnn_hidden_dim = kwargs.get('gnn_hidden_dim', 128)
        embedding_gnn_init(self, node_feature_dim, gnn_hidden_dim, gnn_layers_count)
        for _ in range(gnn_layers_count):
            self.gnn_conv_layers.append(PANConv(gnn_hidden_dim, gnn_hidden_dim, filter_size=4))
    gnn_forward = embedding_gnn_forward

class TransformerPredictor(BaseGNNPredictor):
    def __init__(self, node_feature_dim, edge_feature_dim, **kwargs):
        super().__init__(**kwargs)
        gnn_layers_count = kwargs.get('gnn_layers', 3)
        gnn_hidden_dim = kwargs.get('gnn_hidden_dim', 128)
        dropout_rate = kwargs.get('dropout_rate', 0.3)
        heads = 4
        if gnn_hidden_dim % heads != 0: raise ValueError(f"gnn_hidden_dim ({gnn_hidden_dim}) must be divisible by heads ({heads})")
        embedding_gnn_init(self, node_feature_dim, gnn_hidden_dim, gnn_layers_count)
        self.effective_edge_dim = max(1, edge_feature_dim) if edge_feature_dim > 0 else None
        for _ in range(gnn_layers_count):
            self.gnn_conv_layers.append(
                TransformerConv(in_channels=gnn_hidden_dim, out_channels=gnn_hidden_dim // heads,
                                heads=heads, concat=True, dropout=dropout_rate, edge_dim=self.effective_edge_dim)
            )
    gnn_forward = embedding_gnn_forward