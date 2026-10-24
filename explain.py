"""Reproduce GNNExplainer/Fidelity and fixed-graph SHAP for the paper's 3D GINE models."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import torch
from predict import find_entry,load_model,load_data,indices,context_arrays,FUSIONS,resolve_device
from models import interpretation as I

def select_rows(dat,ck,row_id=None,limit=None):
    ii=indices(ck,dat,'test')
    if row_id is not None:
        ii=np.array([i for i in ii if int(dat['df'].iloc[i].row_id_original)==row_id])
        if not len(ii):raise ValueError('The requested record is not held out in this split.')
    if limit is not None:ii=ii[:limit]
    if not len(ii):raise ValueError('No records selected')
    return ii

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('method',choices=['gnn','shap'])
    ap.add_argument('--fusion',choices=FUSIONS,default='film_concat')
    ap.add_argument('--seed',type=int,choices=range(10),default=0)
    ap.add_argument('--row-id',type=int)
    ap.add_argument('--limit',type=int,help='Maximum held-out records; omit for the complete held-out set')
    ap.add_argument('--device',default='auto')
    ap.add_argument('--output',type=Path,default=Path('outputs/explanations'))
    ap.add_argument('--explainer-seed',type=int,default=0)
    ap.add_argument('--epochs',type=int,default=200)
    ap.add_argument('--nsamples',type=int,default=200)
    ap.add_argument('--species',help='For GNN species-conditioned attribution: replace species only')
    ap.add_argument('--exposure',help='Optional fixed GNN exposure category, e.g. Static')
    ap.add_argument('--duration',type=float,help='Optional fixed GNN exposure duration in days')
    ap.add_argument('--random-repeats',type=int,default=0,help='Optional equally sized random-retention control')
    args=ap.parse_args();torch.set_num_threads(2)
    if args.limit is not None and args.limit<1:ap.error('--limit must be positive')
    if args.random_repeats<0:ap.error('--random-repeats cannot be negative')
    if args.method=='shap' and any(v is not None for v in [args.species,args.exposure,args.duration]):ap.error('Context overrides apply to gnn only')
    if args.epochs<1 or args.nsamples<1:ap.error('epochs/nsamples must be positive')
    device=resolve_device(args.device);ck,model=load_model(find_entry('3D_GINE',args.fusion,args.seed),device)
    dat=load_data();ii=select_rows(dat,ck,args.row_id,args.limit)
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[];atomic=[]
    if args.method=='shap':
        X,nsp,nex=I.build_matrix(dat['df'],dat['moltab'],ck)
        train=indices(ck,dat,'train');test=indices(ck,dat,'test')
        # This matches the paper's original row order and RNG for its TRAIN background.
        bg_idx=np.random.default_rng(91000+args.seed).choice(train,size=100,replace=False)
        bg=X[bg_idx];positions={int(i):j for j,i in enumerate(test)}
        pd.DataFrame({'row_id_original':dat['df'].iloc[bg_idx].row_id_original.to_numpy()}).to_csv(args.output/'background_records.csv',index=False)
    else:xm,xe,sp,ex=context_arrays(ck,dat)
    for i in ii:
        row=dat['df'].iloc[i];rid=int(row.row_id_original);graph=dat['g3d'][i]
        meta={'fusion':args.fusion,'seed':args.seed,'row_id_original':rid,'CAS':str(row.CASNumber1),'species':args.species or row['Species Scientific Name'],'exposure':args.exposure or row['Exposure Type'],'duration_days':args.duration if args.duration is not None else float(row['Exposure Duration'])}
        if args.method=='shap':
            g=I.graph_embedding(model,graph,device)
            sv=I.shap_values(model,g,X[i],bg,nsp,nex,device,92000+args.seed*100000+positions[int(i)],args.nsamples)
            groups=I.grouped(sv,nsp,nex)
            rows.append({**meta,**{'SHAP::'+k:v for k,v in groups.items()},**{'AbsSHAP::'+k:abs(v) for k,v in groups.items()}})
        else:
            species_index=ck['species_classes'].index(args.species) if args.species else int(sp[i])
            exposure_index=ck['exposure_classes'].index(args.exposure) if args.exposure else int(ex[i])
            env=xe[i:i+1] if args.duration is None else I.robust_transform_from_dict(np.array([[args.duration]]),ck['scaler_env'])
            ctx=(torch.tensor(xm[i:i+1],dtype=torch.float32,device=device),torch.tensor(env,dtype=torch.float32,device=device),torch.tensor([species_index],dtype=torch.long,device=device),torch.tensor([exposure_index],dtype=torch.long,device=device))
            wrapper,explanation,orig,x,ei,ea,b=I.run_explainer_once(model,graph,ctx,device,args.explainer_seed,args.epochs)
            scores=torch.as_tensor(I.node_scores(explanation),dtype=torch.float32,device=device)
            n=len(scores);allnodes=torch.arange(n,dtype=torch.long,device=device)
            from rdkit import Chem
            from predict import BASE
            mol=Chem.AddHs(next(iter(Chem.SDMolSupplier(str(BASE/'.cache/structures'/f'{row.Number}.sdf')))),addCoords=True)
            atomic_numbers=np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
            assert len(atomic_numbers)==n
            np.savez_compressed(args.output/f'atom_masks_{rid}.npz',node_mask=explanation.node_mask.detach().cpu().numpy(),edge_mask=explanation.edge_mask.detach().cpu().numpy(),node_scores=scores.cpu().numpy(),edge_index=ei.cpu().numpy(),atomic_numbers=atomic_numbers,heavy_atom_indices=np.flatnonzero(atomic_numbers>1))
            atomic.append({**meta,**I.grouped_atomic_importance(explanation.node_mask.detach().cpu().numpy())})
            if args.random_repeats:
                rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261021,args.seed,rid])))
                permutations=[rng.permutation(n) for _ in range(args.random_repeats)]
            for frac in [.10,.25,.50]:
                k=min(max(1,int(np.ceil(n*frac))),n-1)
                chosen=torch.topk(scores,k=k).indices.sort().values
                mask=torch.zeros(n,dtype=torch.bool,device=device);mask[chosen]=True
                retained=I.pred_subset(wrapper,x,ei,ea,chosen,n);removed=I.pred_subset(wrapper,x,ei,ea,allnodes[~mask],n)
                result={**meta,'fraction':frac,'nodes':n,'retained_nodes':k,'prediction':orig,'retained_prediction':retained,'removed_prediction':removed,'FidelityMinus':abs(orig-retained),'FidelityPlus':abs(orig-removed),'Gini':I.gini(scores.cpu().numpy()),'ShannonEntropy':I.entropy(scores.cpu().numpy()),'selected_atoms':','.join(map(str,chosen.cpu().tolist()))}
                if args.random_repeats:
                    random_values=[]
                    for permutation in permutations:
                        chosen_random=torch.tensor(sorted(permutation[:k]),dtype=torch.long,device=device)
                        random_values.append(abs(orig-I.pred_subset(wrapper,x,ei,ea,chosen_random,n)))
                    result['RandomRetentionFidelityMinus']=float(np.mean(random_values))
                rows.append(result)
        print(f'{args.method}: record {rid}',flush=True)
    pd.DataFrame(rows).to_csv(args.output/f'{args.method}_records.csv',index=False)
    if atomic:pd.DataFrame(atomic).to_csv(args.output/'atomic_features.csv',index=False)
    (args.output/'settings.json').write_text(json.dumps({k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},indent=2),encoding='utf8')

if __name__=='__main__':main()
