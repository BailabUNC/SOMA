import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,json
from pathlib import Path
import torch
from torch import nn
from model import AE32,Encoder64,ae_norm
from train import load_data,fold_indices,seed_all,cpu_state,extract,metric,summarize,save_json
from ablation import fit_head

ROOT=Path(__file__).resolve().parent

class AE64(AE32):
    def __init__(self):
        super().__init__()
        self.enc5=nn.Sequential(nn.Conv1d(16,64,3,2,1),nn.ReLU())
        self.dec5=nn.Sequential(nn.ConvTranspose1d(64,16,3,2,1,output_padding=1),nn.ReLU())

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,default=ROOT/'DATA')
    parser.add_argument('--config',type=Path,default=ROOT/'config.json')
    parser.add_argument('--head-config',type=Path,default=ROOT/'ablation_config.json')
    parser.add_argument('--out',type=Path,default=ROOT/'outputs'/'ae64')
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=parser.parse_args();device=args.device
    cfg=json.loads(args.config.read_text());tc=cfg['training'];hc=json.loads(args.head_config.read_text())
    torch.set_num_threads(cfg['cpu_threads']);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    raw,y,sessions=load_data(args.data_dir,cfg);x=ae_norm(raw).to(device);q=tc['pretrain_quantization'];xx=(x/q).round()*q if q else x
    out=args.out.resolve();out.mkdir(exist_ok=True,parents=True);selections=[]
    for fold in (1,2,3):
        ia,iv,te=fold_indices(y,sessions,fold);seed_all(59+fold*101);ae=AE64().to(device)
        opt=torch.optim.Adam(ae.parameters(),lr=tc['ae_lr']);gen=torch.Generator().manual_seed(1000+fold)
        for ep in range(tc['ae_epochs']):
            ae.train()
            for jj in torch.randperm(len(ia),generator=gen).split(tc['pretrain_batch']):
                b=xx[ia[jj]];loss=torch.nn.functional.smooth_l1_loss(ae(b+.1*torch.randn_like(b)),b)
                opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            if (ep+1)%20==0:print(f'fold{fold} AE64 {ep+1}: {loss.item():.6f}',flush=True)
        en=Encoder64().to(device)
        for j in range(5):en.blocks[j].load_state_dict(getattr(ae,f'enc{j+1}').state_dict())
        en.eval();en.requires_grad_(False)
        with torch.no_grad():z=extract(en,x)
        head,mu,sd,selection=fit_head(z,y,ia,iv,fold,hc,device);selections.append(selection)
        torch.save(dict(ae=cpu_state(ae),encoder=cpu_state(en),classifier=cpu_state(head),mu=mu.cpu(),sd=sd.cpu(),train=ia,val=iv,test=te,val_selection=selection),out/f'fold{fold}.pth')
    rows=[]
    from model import Head
    for fold in (1,2,3):
        ck=torch.load(out/f'fold{fold}.pth',weights_only=False,map_location='cpu');en=Encoder64().to(device);en.load_state_dict(ck['encoder']);en.eval();head=Head().to(device);head.load_state_dict(ck['classifier']);head.eval();te=ck['test']
        with torch.no_grad():pred=head((extract(en,x[te])-ck['mu'].to(device))/ck['sd'].to(device)).argmax(1).cpu().numpy()
        row=metric(y[te].numpy(),pred);rows.append(row);save_json(out/f'fold{fold}_predictions.json',dict(true=y[te].tolist(),pred=pred.tolist(),metrics=row))
    result=dict(folds=rows,summary=summarize(rows),validation=selections);save_json(out/'results.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
