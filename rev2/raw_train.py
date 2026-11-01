import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from model import RawProject
from train import load_data,fold_indices,seed_all,cpu_state,metric,summarize,save_json
from ablation import fit_head

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,default=ROOT/'DATA');p.add_argument('--out',type=Path,default=ROOT/'outputs'/'raw')
    p.add_argument('--config',type=Path,default=ROOT/'config.json');p.add_argument('--head-config',type=Path,default=ROOT/'ablation_config.json')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');args=p.parse_args()
    cfg=json.loads(args.config.read_text());head_cfg=json.loads(args.head_config.read_text());torch.set_num_threads(cfg['cpu_threads'])
    torch.backends.cudnn.allow_tf32=not cfg['training']['full_precision'];torch.backends.cuda.matmul.allow_tf32=False
    x,y,sessions=load_data(args.data_dir,cfg);out=args.out.resolve();out.mkdir(exist_ok=True,parents=True);rows=[]
    for f in (1,2,3):
        ia,iv,te=fold_indices(y,sessions,f);seed_all(59+f*101);en=RawProject().to(args.device)
        raw_mu=x[ia].mean(0,keepdim=True);raw_sd=x[ia].std(0,keepdim=True,unbiased=False);raw_sd=torch.where(raw_sd<1e-8,torch.ones_like(raw_sd),raw_sd)
        with torch.no_grad():z=en(((x-raw_mu)/raw_sd).to(args.device))
        head,mu,sd,selection=fit_head(z,y,ia,iv,f,head_cfg,args.device)
        ck=dict(encoder=cpu_state(en),classifier=cpu_state(head),mu=mu.cpu(),sd=sd.cpu(),raw_mean=raw_mu,raw_std=raw_sd,encoder_kind='raw',train=ia,val=iv,test=te,val_selection=selection)
        torch.save(ck,out/f'fold{f}.pth')
        with torch.no_grad():v=head((en(((x[te]-raw_mu)/raw_sd).to(args.device))-mu)/sd).cpu().numpy()
        pred=v.argmax(1);truth=y[te].numpy();row=metric(truth,pred);rows.append(row)
        save_json(out/f'fold{f}_predictions.json',dict(true=truth.tolist(),pred=pred.tolist(),logits=v.tolist(),original_indices=te.tolist(),confusion=confusion_matrix(truth,pred,labels=np.arange(24)).tolist(),metrics=row))
        print('RAW FRESH',f,row,flush=True)
    save_json(out/'results.json',dict(folds=rows,summary=summarize(rows)));print(json.dumps(summarize(rows)),flush=True)

if __name__=='__main__':main()
