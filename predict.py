"""Load the released LC50 models and predict without fitting or selecting weights."""
from pathlib import Path
import argparse,io,json,zipfile
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from models import gine as G,architectures as A,fusions as F
from models.common import load_atomic_features,robust_transform_from_dict

BASE=Path(__file__).resolve().parent
ARCHITECTURES=['3D_MLP','2D_GINE','3D_GINE','SchNet']
FUSIONS=['concat','film','film_concat']
SCOPES=['main','small','no_species','permute_train']

def entries():
    return json.loads((BASE/'checkpoints/index.json').read_text(encoding='utf8'))

def resolve_device(value='auto'):
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu') if str(value)=='auto' else torch.device(value)

def find_entry(architecture='3D_GINE',fusion='film_concat',seed=0,scope='main'):
    matches=[e for e in entries() if (e['architecture'],e['fusion'],e['seed'],e['scope'])==(architecture,fusion,seed,scope)]
    if len(matches)!=1:raise ValueError('No unique released model for these arguments.')
    return matches[0]

def load_model(entry,device='cpu'):
    device=resolve_device(device)
    with zipfile.ZipFile(BASE/'checkpoints'/entry['archive']) as z:
        ck=torch.load(io.BytesIO(z.read(entry['member'])),map_location='cpu',weights_only=True)
    arch,fusion=ck['architecture'],ck['fusion'];p=ck['best_params'];h=[p['hidden1'],p['hidden2']]
    ns,ne=ck['num_species'],ck['num_exposure']
    if arch in ('2D_GINE','3D_GINE'):
        model=G.make_model_class()(ck['node_dim'],ck['edge_dim'] if arch=='3D_GINE' else 7,3,ns,ne,[p['gnn_dim']]*p['num_layers'],h,p['dropout'],fusion=fusion,use_species=ck['use_species'])
    elif arch=='3D_MLP':
        cls=A.make_mlp_model_class() if fusion=='concat' else F.make_mlp_fusion_model_class()
        model=cls(ck['struct_dim'],3,ns,ne,h,p['dropout'],**({} if fusion=='concat' else {'fusion':fusion}))
    elif arch=='SchNet':
        if device.type=='cpu':warnings.warn('SchNet used CUDA in the study. CPU radius-neighbor selection can differ for crowded neighborhoods. Use --device cuda to reproduce the reported values.')
        cls=A.make_schnet_model_class() if fusion=='concat' else F.make_schnet_fusion_model_class()
        model=cls(3,ns,ne,p['schnet_hidden'],p['num_interactions'],p['num_gaussians'],p['cutoff'],h,p['dropout'],**({} if fusion=='concat' else {'fusion':fusion}))
    else:raise ValueError(arch)
    model.load_state_dict(ck['model_state_dict'],strict=True)
    return ck,model.to(device).eval()

def load_data(dataset='development'):
    folder=BASE/'.cache/structures';folder.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(BASE/'data/structures.zip') as z:
        for info in z.infolist():
            if Path(info.filename).name!=info.filename or not info.filename.endswith('.sdf'):raise ValueError('Invalid structure member')
            dest=folder/info.filename
            if not dest.exists():dest.write_bytes(z.read(info))
    df=pd.read_csv(BASE/'data'/f'{dataset}.csv',dtype={'CASNumber1':str,'Number':str},float_precision='round_trip')
    if not df.row_id_original.is_unique:raise ValueError('Duplicate record IDs')
    if not np.allclose(np.log10(df.lc50_raw),df.lc50_log10,rtol=0,atol=1e-12):raise ValueError('Target scale mismatch')
    return A.prepare_architecture_common_data(df,folder,load_atomic_features(BASE/'data/atomic_features.xlsx'))

def indices(ck,dat,split='test'):
    if split=='all':return np.arange(len(dat['df']))
    return np.flatnonzero(dat['df'].CASNumber1.astype(str).isin(set(map(str,ck['manifest'][split]))))

def context_arrays(ck,dat):
    df=dat['df']
    xm=robust_transform_from_dict(dat['moltab'],ck['scaler_mol'])
    xe=robust_transform_from_dict(df[['Exposure Duration']].to_numpy(float),ck['scaler_env'])
    sp=np.array([ck['species_classes'].index(str(v)) for v in df['Species Scientific Name']])
    ex=np.array([ck['exposure_classes'].index(str(v)) for v in df['Exposure Type']])
    return xm,xe,sp,ex

def predict_model(ck,model,dat,ii,device='cpu',batch_size=None):
    if len(ii)==0:raise ValueError('No records in the requested split')
    if batch_size is None:batch_size=int(ck['best_params']['bs'])
    arch=ck['architecture'];df=dat['df'];xm,xe,sp,ex=context_arrays(ck,dat)
    xm,xe,sp,ex=xm[ii],xe[ii],sp[ii],ex[ii]
    y=df.lc50_log10.to_numpy(float)[ii];ids=df.row_id_original.to_numpy(int)[ii]
    if arch in ('2D_GINE','3D_GINE'):
        ds=G.make_dataset_class()([dat['g2d' if arch=='2D_GINE' else 'g3d'][i] for i in ii],xm,xe,sp,ex,y,ids);collate=G.collate_fn;ev=G.evaluate
    elif arch=='3D_MLP':
        xs=robust_transform_from_dict(dat['struct'],ck['scaler_struct'])[ii]
        ds=A.make_mlp_dataset_class()(xs,xm,xe,sp,ex,y,ids);collate=None;ev=A.evaluate_mlp
    else:
        ds=A.make_schnet_dataset_class()([dat['schnet'][i] for i in ii],xm,xe,sp,ex,y,ids);collate=A.schnet_collate;ev=A.evaluate_schnet
    with torch.no_grad():
        _,pred,true,rids=ev(model,DataLoader(ds,batch_size=batch_size,shuffle=False,collate_fn=collate),torch.nn.MSELoss(),torch.device(device))
    out=pd.DataFrame({'row_id_original':np.asarray(rids).flatten(),'True_log10':np.asarray(true).flatten(),'Pred_log10':np.asarray(pred).flatten()})
    meta=df.set_index('row_id_original')
    out.insert(1,'CAS',meta.loc[out.row_id_original,'CASNumber1'].to_numpy())
    return out

def metrics(out):
    from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
    return {'R2':float(r2_score(out.True_log10,out.Pred_log10)),'RMSE':float(np.sqrt(mean_squared_error(out.True_log10,out.Pred_log10))),'MAE':float(mean_absolute_error(out.True_log10,out.Pred_log10))}

def add_model_arguments(ap):
    ap.add_argument('--architecture',choices=ARCHITECTURES,default='3D_GINE')
    ap.add_argument('--fusion',choices=FUSIONS,default='film_concat')
    ap.add_argument('--seed',type=int,choices=range(10),default=0)
    ap.add_argument('--scope',choices=SCOPES,default='main')
    ap.add_argument('--device',default='auto')
    ap.add_argument('--output',type=Path,default=Path('outputs/predictions.csv'))

def main():
    ap=argparse.ArgumentParser(description=__doc__);add_model_arguments(ap)
    ap.add_argument('--dataset',choices=['development','external'],default='development')
    ap.add_argument('--split',choices=['train','val','test'],default='test')
    ap.add_argument('--all',action='store_true',help='Evaluate every released checkpoint in the selected scope')
    a=ap.parse_args();a.device=resolve_device(a.device);torch.set_num_threads(2)
    dat=load_data(a.dataset)
    selected=[e for e in entries() if e['scope']==a.scope] if a.all else [find_entry(a.architecture,a.fusion,a.seed,a.scope)]
    a.output.parent.mkdir(parents=True,exist_ok=True);reports=[]
    for e in selected:
        ck,model=load_model(e,a.device)
        if a.dataset=='external' and set(dat['df'].CASNumber1)&set(sum(ck['manifest'].values(),[])):raise ValueError('External chemical overlaps development')
        ii=indices(ck,dat,'all' if a.dataset=='external' else a.split)
        out=predict_model(ck,model,dat,ii,a.device)
        rep={k:e[k] for k in ['scope','architecture','fusion','seed']};rep.update(records=len(out),**metrics(out));reports.append(rep)
        print(json.dumps(rep),flush=True)
        if not a.all:out.to_csv(a.output,index=False)
    if a.all:pd.DataFrame(reports).to_csv(a.output,index=False)

if __name__=='__main__':main()
