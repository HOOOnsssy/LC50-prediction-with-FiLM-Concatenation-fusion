from __future__ import annotations
from typing import Dict,List,Optional,Sequence,Tuple
import numpy as np
import pandas as pd
from .common import ALLOWED_ELEMENTS,set_seed,safe_r2,rmse

def one_hot_encoding(value, choices):
    encoding = [0] * len(choices)
    if value in choices:
        encoding[choices.index(value)] = 1
    else:
        encoding[-1] = 1
    return encoding


def build_graph_from_rdkit(mol, atomic_df):
    """Reviewer-corrected 3D-GINE graph construction, preserving the manuscript feature design.

    Node features: 10 atomic properties + Gasteiger + hybridization + degree +
    formal charge + aromatic + ring + H count.
    Edge features: bond type + ring + conjugation + 3D bond length (8 dims).

    Important correction: SDFs are read with RDKit's standard removeHs=True behavior,
    then explicit H atoms are re-added with addCoords=True before any H-bond distance
    is calculated. This keeps the original graph topology while avoiding origin/undefined
    coordinates for newly added hydrogen atoms.
    """
    import torch
    from rdkit import Chem
    from rdkit.Chem import rdPartialCharges

    if mol is None:
        return None
    mol = Chem.AddHs(mol, addCoords=True)
    try:
        rdPartialCharges.ComputeGasteigerCharges(mol)
    except Exception:
        pass

    try:
        conf = mol.GetConformer()
        has_3d = True
    except ValueError:
        conf = None
        has_3d = False

    node_features = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol not in ALLOWED_ELEMENTS:
            return None
        atom_feat = atomic_df.loc[symbol].values.tolist() if symbol in atomic_df.index else np.zeros(atomic_df.shape[1]).tolist()
        try:
            g_charge = atom.GetDoubleProp('_GasteigerCharge')
            if np.isnan(g_charge) or np.isinf(g_charge):
                g_charge = 0.0
        except Exception:
            g_charge = 0.0

        hybridization = one_hot_encoding(str(atom.GetHybridization()), ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'Unknown'])
        degree = one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 'Unknown'])
        formal_charge = [atom.GetFormalCharge()]
        aromatic = [1 if atom.GetIsAromatic() else 0]
        in_ring = [1 if atom.IsInRing() else 0]
        num_hs = one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 'Unknown'])
        node_features.append(atom_feat + [g_charge] + hybridization + degree + formal_charge + aromatic + in_ring + num_hs)

    x = torch.tensor(node_features, dtype=torch.float32)
    edge_index, edge_features = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = one_hot_encoding(str(bond.GetBondType()), ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC', 'Unknown'])
        bond_in_ring = [1 if bond.IsInRing() else 0]
        conj = [1 if bond.GetIsConjugated() else 0]
        if has_3d:
            try:
                pi, pj = conf.GetAtomPosition(i), conf.GetAtomPosition(j)
                dist = float(np.sqrt((pi.x-pj.x)**2 + (pi.y-pj.y)**2 + (pi.z-pj.z)**2))
                if not np.isfinite(dist) or dist > 10.0 or dist < 0.1:
                    dist = 1.5
            except Exception:
                dist = 1.5
        else:
            dist = 1.5
        ef = bond_type + bond_in_ring + conj + [dist]
        edge_index += [[i, j], [j, i]]
        edge_features += [ef, ef]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.zeros((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(edge_features, dtype=torch.float32) if edge_features else torch.zeros((0, 8), dtype=torch.float32)
    if torch.isnan(x).any() or torch.isnan(edge_attr).any():
        return None
    return {'x': x, 'edge_index': edge_index, 'edge_attr': edge_attr}


def get_torch_components():
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import Dataset
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool, GraphNorm
    return torch, F, nn, Dataset, Data, Batch, GINEConv, global_mean_pool, global_max_pool, GraphNorm


def make_model_class():
    torch, F, nn, Dataset, Data, Batch, GINEConv, global_mean_pool, global_max_pool, GraphNorm = get_torch_components()

    class MultiBranchGINEModel(nn.Module):
        def __init__(self, node_dim, edge_dim, mol_dim, num_species, num_exposure,
                     gnn_dims, hidden_dims, dropout, fusion='film_concat', use_species=True):
            super().__init__()
            if fusion == 'film_residual':
                fusion = 'film_concat'  # backward-compatible alias
            if fusion not in {'concat', 'film', 'film_concat'}:
                raise ValueError('fusion must be concat, film, or film_concat')
            self.fusion = fusion
            self.use_species = use_species
            self.gnn_layers = nn.ModuleList()
            self.graph_norms = nn.ModuleList()
            self.edge_encoders = nn.ModuleList()
            in_dim = node_dim
            for out_dim in gnn_dims:
                mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))
                self.gnn_layers.append(GINEConv(nn=mlp, edge_dim=out_dim))
                self.edge_encoders.append(nn.Linear(edge_dim, out_dim))
                self.graph_norms.append(GraphNorm(out_dim))
                in_dim = out_dim

            self.dropout = nn.Dropout(dropout)
            gnn_out_dim = gnn_dims[-1] * 2
            h1, h2 = hidden_dims
            self.fc_graph = nn.Sequential(nn.Linear(gnn_out_dim, h1), nn.BatchNorm1d(h1), nn.SiLU())
            self.fc_mol = nn.Sequential(nn.Linear(mol_dim, 16), nn.BatchNorm1d(16), nn.SiLU(), nn.Linear(16, 16), nn.SiLU())

            emb_dim_sp = min(50, (num_species + 1) // 2) if use_species else 0
            emb_dim_ex = min(10, (num_exposure + 1) // 2)
            self.emb_species = nn.Embedding(num_species, emb_dim_sp) if use_species else None
            self.emb_exposure = nn.Embedding(num_exposure, emb_dim_ex)
            env_dim = 1 + emb_dim_ex + emb_dim_sp
            self.fc_env = nn.Sequential(nn.Linear(env_dim, h1), nn.BatchNorm1d(h1), nn.SiLU())

            if fusion in {'film', 'film_concat'}:
                self.cond_proj_m = nn.Sequential(nn.Linear(16, h1), nn.BatchNorm1d(h1), nn.SiLU())
                self.film_gamma_m = nn.Linear(h1, h1)
                self.film_beta_m = nn.Linear(h1, h1)
                self.cond_proj_e = nn.Sequential(nn.Linear(h1, h1), nn.BatchNorm1d(h1), nn.SiLU())
                self.film_gamma_e = nn.Linear(h1, h1)
                self.film_beta_e = nn.Linear(h1, h1)

            if fusion == 'concat':
                combined_dim = h1 + 16 + h1
            elif fusion == 'film':
                combined_dim = h1
            else:
                combined_dim = h1 + 16 + h1
            self.fc_joint = nn.Sequential(nn.Linear(combined_dim, h2), nn.BatchNorm1d(h2), nn.SiLU())
            self.fc_out = nn.Linear(h2, 1)

        def forward(self, x, edge_index, edge_attr, batch, tab_mol, tab_env_cont, tab_env_sp, tab_env_ex):
            for gnn, edge_enc, norm in zip(self.gnn_layers, self.edge_encoders, self.graph_norms):
                eedge = F.relu(edge_enc(edge_attr))
                x = F.relu(norm(gnn(x, edge_index, edge_attr=eedge), batch))
            g = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1)
            g = self.dropout(self.fc_graph(g))
            m = self.dropout(self.fc_mol(tab_mol))

            pieces = [tab_env_cont]
            if self.use_species:
                pieces.append(self.emb_species(tab_env_sp.view(-1)))
            pieces.append(self.emb_exposure(tab_env_ex.view(-1)))
            e = self.dropout(self.fc_env(torch.cat(pieces, dim=1)))

            if self.fusion == 'concat':
                combined = torch.cat([g, m, e], dim=1)
            else:
                cm = self.cond_proj_m(m)
                g_mol = (1 + torch.tanh(self.film_gamma_m(cm))) * g + self.film_beta_m(cm)
                ce = self.cond_proj_e(e)
                g_final = (1 + torch.tanh(self.film_gamma_e(ce))) * g_mol + self.film_beta_e(ce)
                combined = g_final if self.fusion == 'film' else torch.cat([g_final, m, e], dim=1)
            return self.fc_out(self.fc_joint(combined)).view(-1)

    return MultiBranchGINEModel


def make_dataset_class():
    torch, F, nn, Dataset, Data, Batch, GINEConv, global_mean_pool, global_max_pool, GraphNorm = get_torch_components()
    class MolecularDataset(Dataset):
        def __init__(self, graphs, tab_mol, tab_env_cont, tab_env_sp, tab_env_ex, y, row_ids=None):
            self.data_list = []
            if row_ids is None:
                row_ids = np.arange(len(y))
            for g, tm, tc, tsp, tex, target, rid in zip(graphs, tab_mol, tab_env_cont, tab_env_sp, tab_env_ex, y, row_ids):
                self.data_list.append(Data(
                    x=g['x'], edge_index=g['edge_index'], edge_attr=g['edge_attr'],
                    tab_mol=torch.tensor(tm, dtype=torch.float32).unsqueeze(0),
                    tab_env_cont=torch.tensor(tc, dtype=torch.float32).unsqueeze(0),
                    tab_env_sp=torch.tensor([tsp], dtype=torch.long),
                    tab_env_ex=torch.tensor([tex], dtype=torch.long),
                    y=torch.tensor(target, dtype=torch.float32),
                    row_id=torch.tensor([int(rid)], dtype=torch.long),
                ))
        def __len__(self): return len(self.data_list)
        def __getitem__(self, idx): return self.data_list[idx]
    return MolecularDataset


def collate_fn(batch):
    from torch_geometric.data import Batch
    return Batch.from_data_list(batch)


def evaluate(model, loader, criterion, device):
    import torch
    model.eval()
    preds, trues, row_ids, total_loss = [], [], [], 0.0
    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            pred = model(b.x, b.edge_index, b.edge_attr, b.batch, b.tab_mol, b.tab_env_cont, b.tab_env_sp, b.tab_env_ex)
            total_loss += float(criterion(pred, b.y).item())
            preds.extend(pred.detach().cpu().numpy().tolist())
            trues.extend(b.y.detach().cpu().numpy().tolist())
            row_ids.extend(b.row_id.detach().cpu().numpy().reshape(-1).tolist())
    return total_loss / max(len(loader), 1), np.asarray(preds), np.asarray(trues), np.asarray(row_ids, dtype=int)
