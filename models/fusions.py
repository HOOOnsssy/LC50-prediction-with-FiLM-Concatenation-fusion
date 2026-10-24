from __future__ import annotations

def make_mlp_fusion_model_class():
    """Exact revised analogue of the original MLP FiLM / FiLM-Concat models."""
    import torch
    from torch import nn

    class MultiBranchMLPFusion(nn.Module):
        def __init__(self, dim_struct, dim_mol, num_species, num_exposure,
                     hidden_dims, dropout, fusion='film'):
            super().__init__()
            if fusion not in {'film', 'film_concat'}:
                raise ValueError('fusion must be film or film_concat')
            self.fusion = fusion
            h1, h2 = hidden_dims

            self.fc_struct = nn.Sequential(
                nn.Linear(dim_struct, 16), nn.BatchNorm1d(16), nn.SiLU(),
                nn.Linear(16, h1), nn.SiLU(),
            )
            self.fc_mol = nn.Sequential(
                nn.Linear(dim_mol, 16), nn.BatchNorm1d(16), nn.SiLU(),
                nn.Linear(16, 16), nn.SiLU(),
            )
            emb_dim_sp = min(50, (num_species + 1) // 2)
            emb_dim_ex = min(10, (num_exposure + 1) // 2)
            self.emb_species = nn.Embedding(num_species, emb_dim_sp)
            self.emb_exposure = nn.Embedding(num_exposure, emb_dim_ex)
            self.dropout = nn.Dropout(dropout)
            self.fc_env = nn.Sequential(
                nn.Linear(1 + emb_dim_sp + emb_dim_ex, h1),
                nn.BatchNorm1d(h1), nn.SiLU(),
            )

            self.cond_proj_m = nn.Sequential(nn.Linear(16, h1), nn.BatchNorm1d(h1), nn.SiLU())
            self.film_gamma_m = nn.Linear(h1, h1)
            self.film_beta_m = nn.Linear(h1, h1)
            self.cond_proj_e = nn.Sequential(nn.Linear(h1, h1), nn.BatchNorm1d(h1), nn.SiLU())
            self.film_gamma_e = nn.Linear(h1, h1)
            self.film_beta_e = nn.Linear(h1, h1)

            combined_dim = h1 if fusion == 'film' else (h1 + 16 + h1)
            self.fc_joint = nn.Sequential(nn.Linear(combined_dim, h2), nn.BatchNorm1d(h2), nn.SiLU())
            self.fc_out = nn.Linear(h2, 1)

        def forward(self, x_struct, x_mol, x_env_cont, x_env_sp, x_env_ex):
            s = self.dropout(self.fc_struct(x_struct))
            m = self.dropout(self.fc_mol(x_mol))
            sp = self.emb_species(x_env_sp.view(-1))
            ex = self.emb_exposure(x_env_ex.view(-1))
            e = self.dropout(self.fc_env(torch.cat([x_env_cont, sp, ex], dim=1)))

            cm = self.cond_proj_m(m)
            s_mol = (1 + torch.tanh(self.film_gamma_m(cm))) * s + self.film_beta_m(cm)
            ce = self.cond_proj_e(e)
            s_final = (1 + torch.tanh(self.film_gamma_e(ce))) * s_mol + self.film_beta_e(ce)

            combined = s_final if self.fusion == 'film' else torch.cat([s_final, m, e], dim=1)
            return self.fc_out(self.fc_joint(combined)).view(-1)

    return MultiBranchMLPFusion


def make_schnet_fusion_model_class():
    """Corrected-coordinate SchNet with the same hierarchical FiLM definitions as the original SI."""
    import torch
    from torch import nn
    from torch_geometric.nn import SchNet, global_max_pool

    class MultiBranchSchNetFusion(nn.Module):
        def __init__(self, mol_dim, num_species, num_exposure, schnet_hidden,
                     num_interactions, num_gaussians, cutoff,
                     hidden_dims, dropout, fusion='film'):
            super().__init__()
            if fusion not in {'film', 'film_concat'}:
                raise ValueError('fusion must be film or film_concat')
            self.fusion = fusion

            self.schnet = SchNet(
                hidden_channels=schnet_hidden,
                num_filters=schnet_hidden,
                num_interactions=num_interactions,
                num_gaussians=num_gaussians,
                cutoff=cutoff,
                readout='mean',
            )
            # Match script 13/original study: intercept the pre-output node representation,
            # then combine mean and max graph readouts.
            self.schnet.lin2 = nn.Identity()
            self._node_features = None

            def hook_fn(module, inputs, output):
                self._node_features = output

            self.schnet.lin2.register_forward_hook(hook_fn)

            h1, h2 = hidden_dims
            schnet_out_dim = (schnet_hidden // 2) * 2
            self.dropout = nn.Dropout(dropout)
            self.fc_graph = nn.Sequential(nn.Linear(schnet_out_dim, h1), nn.BatchNorm1d(h1), nn.SiLU())
            self.fc_mol = nn.Sequential(
                nn.Linear(mol_dim, 16), nn.BatchNorm1d(16), nn.SiLU(),
                nn.Linear(16, 16), nn.SiLU(),
            )
            emb_dim_sp = min(50, (num_species + 1) // 2)
            emb_dim_ex = min(10, (num_exposure + 1) // 2)
            self.emb_species = nn.Embedding(num_species, emb_dim_sp)
            self.emb_exposure = nn.Embedding(num_exposure, emb_dim_ex)
            self.fc_env = nn.Sequential(
                nn.Linear(1 + emb_dim_sp + emb_dim_ex, h1),
                nn.BatchNorm1d(h1), nn.SiLU(),
            )

            self.cond_proj_m = nn.Sequential(nn.Linear(16, h1), nn.BatchNorm1d(h1), nn.SiLU())
            self.film_gamma_m = nn.Linear(h1, h1)
            self.film_beta_m = nn.Linear(h1, h1)
            self.cond_proj_e = nn.Sequential(nn.Linear(h1, h1), nn.BatchNorm1d(h1), nn.SiLU())
            self.film_gamma_e = nn.Linear(h1, h1)
            self.film_beta_e = nn.Linear(h1, h1)

            combined_dim = h1 if fusion == 'film' else (h1 + 16 + h1)
            self.fc_joint = nn.Sequential(nn.Linear(combined_dim, h2), nn.BatchNorm1d(h2), nn.SiLU())
            self.fc_out = nn.Linear(h2, 1)

        def forward(self, z, pos, batch, tab_mol, tab_env_cont, tab_env_sp, tab_env_ex):
            g_mean = self.schnet(z, pos, batch)
            h = self._node_features
            if h is None:
                raise RuntimeError('SchNet node-feature hook did not fire.')
            g_max = global_max_pool(h, batch)
            g = self.dropout(self.fc_graph(torch.cat([g_mean, g_max], dim=-1)))

            m = self.dropout(self.fc_mol(tab_mol))
            sp = self.emb_species(tab_env_sp.view(-1))
            ex = self.emb_exposure(tab_env_ex.view(-1))
            e = self.dropout(self.fc_env(torch.cat([tab_env_cont, sp, ex], dim=1)))

            cm = self.cond_proj_m(m)
            g_mol = (1 + torch.tanh(self.film_gamma_m(cm))) * g + self.film_beta_m(cm)
            ce = self.cond_proj_e(e)
            g_final = (1 + torch.tanh(self.film_gamma_e(ce))) * g_mol + self.film_beta_e(ce)

            combined = g_final if self.fusion == 'film' else torch.cat([g_final, m, e], dim=1)
            return self.fc_out(self.fc_joint(combined)).view(-1)

    return MultiBranchSchNetFusion
