from __future__ import annotations
import numpy as np
import pandas as pd
from models.common import set_seed,robust_transform_from_dict

FEATURE_GROUPS = [
    ("Group Num", [0]),
    ("Period Num", [1]),
    ("Electronegativity", [2]),
    ("Covalent Radii", [3]),
    ("Valence Elec", [4]),
    ("Ionization Eng", [5]),
    ("Electron Aff", [6]),
    ("Density", [7]),
    ("Atomic Mass", [8]),
    ("Atomic Vol", [9]),
    ("Gasteiger", [10]),
    ("Hybridization", list(range(11, 17))),
    ("Degree", list(range(17, 25))),
    ("Formal Charge", [25]),
    ("Aromatic", [26]),
    ("In Ring", [27]),
    ("Num_Hs", list(range(28, 34))),
]

def run_explainer_once(model, graph: dict, context, device, explainer_seed: int | None = None, epochs: int = 200):
    import torch
    from torch_geometric.explain import Explainer, GNNExplainer

    tm, tc, tsp, tex = context

    class Wrapper(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
        def forward(self, x, edge_index, edge_attr, batch):
            return self.base(x, edge_index, edge_attr, batch, tm, tc, tsp, tex)

    wrapper = Wrapper(model).to(device)
    wrapper.eval()
    x = graph["x"].to(device)
    ei = graph["edge_index"].to(device)
    ea = graph["edge_attr"].to(device)
    batch = torch.zeros(x.size(0), dtype=torch.long, device=device)

    if explainer_seed is not None:
        set_seed(int(explainer_seed))

    explainer = Explainer(
        model=wrapper,
        algorithm=GNNExplainer(epochs=int(epochs)),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(mode="regression", task_level="graph", return_type="raw"),
    )
    exp = explainer(x=x, edge_index=ei, edge_attr=ea, batch=batch)
    if exp.node_mask is None:
        raise RuntimeError("GNNExplainer returned no node_mask")
    with torch.no_grad():
        pred = float(wrapper(x, ei, ea, batch).item())
    return wrapper, exp, pred, x, ei, ea, batch

def grouped_atomic_importance(node_mask) -> dict[str, float]:
    arr = np.asarray(node_mask, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D node mask, got {arr.shape}")
    raw = np.abs(arr).mean(axis=0)
    if len(raw) != 34:
        raise ValueError(f"Expected 34 node-feature dimensions, got {len(raw)}")
    return {label: float(np.sum(raw[idxs])) for label, idxs in FEATURE_GROUPS}

def node_scores(exp) -> np.ndarray:
    mask = exp.node_mask.detach().abs()
    if mask.ndim == 1:
        v = mask
    else:
        v = mask.mean(dim=1)
    return v.cpu().numpy().astype(float)

def gini(values) -> float:
    x = np.asarray(values, dtype=float).reshape(-1)
    x = np.abs(x[np.isfinite(x)])
    if len(x) == 0 or x.sum() <= 1e-15:
        return 0.0
    x = np.sort(x)
    n = len(x)
    i = np.arange(1, n + 1, dtype=float)
    return float(np.sum((2 * i - n - 1) * x) / (n * x.sum()))

def entropy(values) -> float:
    x = np.asarray(values, dtype=float).reshape(-1)
    x = np.abs(x[np.isfinite(x)])
    if len(x) == 0 or x.sum() <= 1e-15:
        return 0.0
    p = x / x.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

def top_fraction_set(scores, fraction=.25) -> set[int]:
    scores = np.asarray(scores, dtype=float)
    k = max(1, int(np.ceil(len(scores) * float(fraction))))
    return set(np.argsort(scores)[-k:].tolist())

def safe_spearman(a, b) -> float:
    from scipy.stats import spearmanr
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 2 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return np.nan
    return float(spearmanr(a, b).statistic)

def pair_metrics(a, b, fraction=.25) -> dict[str, float]:
    a = np.asarray(a, float); b = np.asarray(b, float)
    sa = top_fraction_set(a, fraction); sb = top_fraction_set(b, fraction)
    return {
        "Jaccard": len(sa & sb) / max(len(sa | sb), 1),
        "Spearman": safe_spearman(a, b),
        "MeanAbsDifference": float(np.mean(np.abs(a - b))),
    }

def pred_subset(wrapper,x,ei,ea,nodes,num_nodes):
    import torch
    from torch_geometric.utils import subgraph
    if int(nodes.numel())==0: return np.nan
    sei,sea=subgraph(nodes,ei,edge_attr=ea,relabel_nodes=True,num_nodes=num_nodes)
    sx=x[nodes]; sb=torch.zeros(sx.size(0),dtype=torch.long,device=x.device)
    with torch.no_grad(): return float(wrapper(sx,sei,sea,sb).item())

def build_matrix(df,X_mol,ck):
    mol=robust_transform_from_dict(X_mol,ck["scaler_mol"])
    env=robust_transform_from_dict(df[["Exposure Duration"]].astype(float).values,ck["scaler_env"])
    spc=list(ck["species_classes"]); exc=list(ck["exposure_classes"])
    sm={x:i for i,x in enumerate(spc)}; em={x:i for i,x in enumerate(exc)}
    sp=np.asarray([sm[str(x)] for x in df["Species Scientific Name"]],int)
    ex=np.asarray([em[str(x)] for x in df["Exposure Type"]],int)
    X=np.hstack([mol,env,np.eye(len(spc),dtype=np.float32)[sp],np.eye(len(exc),dtype=np.float32)[ex]]).astype(np.float32)
    return X,len(spc),len(exc)

def graph_embedding(model,graph,device):
    import torch, torch.nn.functional as F
    from torch_geometric.nn import global_mean_pool,global_max_pool
    x=graph["x"].to(device); ei=graph["edge_index"].to(device); ea=graph["edge_attr"].to(device)
    b=torch.zeros(x.size(0),dtype=torch.long,device=device)
    model.eval()
    with torch.no_grad():
        h=x
        for gnn,edge_enc,norm in zip(model.gnn_layers,model.edge_encoders,model.graph_norms):
            e=F.relu(edge_enc(ea)); h=F.relu(norm(gnn(h,ei,edge_attr=e),b))
        g=torch.cat([global_mean_pool(h,b),global_max_pool(h,b)],dim=-1)
        return model.dropout(model.fc_graph(g)).detach()

def shap_values(model,g,target,bg,nsp,nex,device,seed,nsamples=200):
    import torch, shap
    class W(torch.nn.Module):
        def __init__(self,base,g): super().__init__(); self.base=base; self.register_buffer("g",g.clone())
        def forward(self,z):
            n=z.shape[0]; g=self.g.repeat(n,1)
            tm=z[:,:3]; tc=z[:,3:4]; so=z[:,4:4+nsp]; eo=z[:,4+nsp:4+nsp+nex]
            m=self.base.dropout(self.base.fc_mol(tm))
            se=torch.matmul(so,self.base.emb_species.weight) if self.base.use_species else torch.zeros((n,0),dtype=z.dtype,device=z.device)
            ee=torch.matmul(eo,self.base.emb_exposure.weight)
            e=self.base.dropout(self.base.fc_env(torch.cat([tc,se,ee],dim=1)))
            if self.base.fusion=="concat": comb=torch.cat([g,m,e],dim=1)
            else:
                cm=self.base.cond_proj_m(m); gm=(1+torch.tanh(self.base.film_gamma_m(cm)))*g+self.base.film_beta_m(cm)
                ce=self.base.cond_proj_e(e); gf=(1+torch.tanh(self.base.film_gamma_e(ce)))*gm+self.base.film_beta_e(ce)
                comb=gf if self.base.fusion=="film" else torch.cat([gf,m,e],dim=1)
            return self.base.fc_out(self.base.fc_joint(comb)).view(-1,1)
    w=W(model,g).to(device); w.eval()
    bt=torch.tensor(bg,dtype=torch.float32,device=device); tt=torch.tensor(target.reshape(1,-1),dtype=torch.float32,device=device)
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    ex=shap.GradientExplainer(w,bt)
    try: sv=ex.shap_values(tt,nsamples=nsamples,rseed=seed)
    except TypeError: sv=ex.shap_values(tt,nsamples=nsamples)
    if isinstance(sv,list): sv=sv[0]
    a=np.asarray(sv,float).squeeze().reshape(-1)
    if len(a)!=len(target): raise RuntimeError(f"Unexpected SHAP shape {np.asarray(sv).shape}")
    return a

def grouped(sv,nsp,nex):
    s0=4; s1=s0+nsp; e0=s1; e1=e0+nex
    return {
        "MolWt":float(sv[0]),"TPSA":float(sv[1]),"LogP":float(sv[2]),"Exposure Duration":float(sv[3]),
        "Scientific Name":float(np.sum(sv[s0:s1])),"Exposure Type":float(np.sum(sv[e0:e1])),
    }
