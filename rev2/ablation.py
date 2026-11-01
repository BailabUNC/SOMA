import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse,json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from model import AE32,Encoder64,RawProject,Projector,Predictor,Head,ae_norm,augment,augment_table_s4
from train import load_data,fold_indices,seed_all,cpu_state,extract,save_json,metric,summarize

ROOT=Path(__file__).resolve().parent

def pretrain(ae,raw,fold,kind,config,device):
    seed_all(59+fold*101)
    en=Encoder64(ae,'new').to(device)
    freeze=config['lastblock_freeze_first_blocks'] if kind=='lastblock_ssl' else config['freeze_first_blocks']
    for j,b in enumerate(en.blocks):
        for p in b.parameters():p.requires_grad=j>=freeze
    pr=Projector().to(device);pd=Predictor().to(device)
    opt=torch.optim.Adam([p for p in en.parameters() if p.requires_grad]+list(pr.parameters())+list(pd.parameters()),lr=config['ssl_lr'])
    x=ae_norm(raw).to(device);q=config['pretrain_quantization']
    if q>0:x=(x/q).round()*q
    gen=torch.Generator().manual_seed(20000+fold)
    for ep in range(config['ssl_epochs']):
        en.train();pr.train();pd.train()
        for ix in torch.randperm(len(x),generator=gen).split(config['batch_size']):
            xb=x[ix]
            if kind=='s4_ssl':v1=augment_table_s4(xb);v2=augment_table_s4(xb)
            elif kind=='mild_ssl':
                v1=augment(xb,config['mild_noise'],config['mild_channel_drop'],config['mild_scale'])
                v2=augment(xb,config['mild_noise'],config['mild_channel_drop'],config['mild_scale'])
            else:
                v1=augment(xb,config['default_noise'],config['default_channel_drop'],config['default_scale'])
                v2=augment(xb,config['default_noise'],config['default_channel_drop'],config['default_scale'])
            z1=pr(en(v1));z2=pr(en(v2));p1=pd(z1);p2=pd(z2)
            loss=-.5*(torch.nn.functional.cosine_similarity(p1,z2.detach()).mean()+torch.nn.functional.cosine_similarity(p2,z1.detach()).mean())
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if (ep+1)%25==0:print(f'fold{fold} {kind} SSL {ep+1}: {loss.item():.6f}',flush=True)
    return en,pr,pd

def fit_head(z,y,ia,iv,fold,config,device):
    mu=z[ia].mean(0);sd=z[ia].std(0,unbiased=False).clamp(min=1e-6);z=(z-mu)/sd
    yt=y.to(device);seed_all(5011+(fold-1)*113);head=Head().to(device)
    opt=torch.optim.AdamW(head.parameters(),lr=config['head_lr'],weight_decay=config['head_weight_decay'])
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=config['head_epochs'])
    gen=torch.Generator().manual_seed(3000+fold-1);best=(-1,float('inf'));state=None;epoch=0
    for ep in range(config['head_epochs']):
        head.train()
        for jj in torch.randperm(len(ia),generator=gen).split(config['batch_size']):
            ix=ia[jj];loss=torch.nn.functional.cross_entropy(head(z[ix]),yt[ix])
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        sch.step();head.eval()
        with torch.no_grad():
            logits=head(z[iv]);acc=int((logits.argmax(1)==yt[iv]).sum());ce=float(torch.nn.functional.cross_entropy(logits,yt[iv]))
        if acc>best[0] or (acc==best[0] and ce<best[1]):best=(acc,ce);state=cpu_state(head);epoch=ep+1
    head.load_state_dict(state);head.eval()
    return head,mu,sd,dict(val_correct=best[0],val_ce=best[1],epoch=epoch)

def main():
    p=argparse.ArgumentParser();p.add_argument('--models-dir',type=Path,default=ROOT/'outputs'/'main');p.add_argument('--out',type=Path,default=ROOT/'outputs'/'augmentation')
    p.add_argument('--data-dir',type=Path,default=ROOT/'DATA');p.add_argument('--config',type=Path,default=ROOT/'config.json');p.add_argument('--head-config',type=Path,default=ROOT/'ablation_config.json')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');p.add_argument('--variants',nargs='+');args=p.parse_args()
    config=json.loads(args.config.read_text());cfg=json.loads(args.head_config.read_text())
    kinds=args.variants or cfg['variants'];torch.set_num_threads(config['cpu_threads'])
    torch.backends.cudnn.allow_tf32=not config['training']['full_precision'];torch.backends.cuda.matmul.allow_tf32=False
    x,y,sessions=load_data(args.data_dir,config);out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    save_json(out/'config.json',cfg);report={'folds':{},'selected':{}}
    for fold in (1,2,3):
        reference=torch.load(args.models_dir/f'fold{fold}.pth',weights_only=False,map_location='cpu')
        ia,iv,te=fold_indices(y,sessions,fold);directory=out/f'fold{fold}';directory.mkdir(exist_ok=True)
        report['folds'][str(fold)]={}
        for kind in kinds:
            print(f'fold{fold} {kind}',flush=True);seed_all(59+fold*101);extra={}
            if kind=='raw':
                en=RawProject().to(args.device);raw_mu=x[ia].mean(0,keepdim=True);raw_sd=x[ia].std(0,keepdim=True,unbiased=False)
                raw_sd=torch.where(raw_sd<1e-8,torch.ones_like(raw_sd),raw_sd)
                with torch.no_grad():z=en(((x-raw_mu)/raw_sd).to(args.device))
                extra=dict(raw_mean=raw_mu,raw_std=raw_sd,encoder_kind='raw')
            elif kind in cfg['variants']:
                ae=AE32().to(args.device);ae.load_state_dict(reference['ae'])
                if kind=='saved_ssl':en=Encoder64().to(args.device);en.load_state_dict(reference['encoder'])
                elif kind=='ae_only':seed_all(59+fold*101);en=Encoder64(ae,'new').to(args.device)
                else:
                    en,pr,pd=pretrain(ae,x[ia],fold,kind,cfg,args.device)
                    extra=dict(projector=cpu_state(pr),predictor=cpu_state(pd))
                en.eval();en.requires_grad_(False)
                with torch.no_grad():z=extract(en,ae_norm(x).to(args.device))
            else:raise ValueError(kind)
            en.eval();en.requires_grad_(False)
            head,mu,sd,selection=fit_head(z,y,ia,iv,fold,cfg,args.device)
            checkpoint=dict(encoder=cpu_state(en),classifier=cpu_state(head),mu=mu.cpu(),sd=sd.cpu(),train=ia,val=iv,test=te,variant=kind,val_selection=selection,**extra)
            torch.save(checkpoint,directory/f'{kind}.pth')
            report['folds'][str(fold)][kind]={'validation':selection}
        report['selected'][str(fold)]=min(kinds,key=lambda kind:(-report['folds'][str(fold)][kind]['validation']['val_correct'],report['folds'][str(fold)][kind]['validation']['val_ce'],kinds.index(kind)))
    for fold in (1,2,3):
        for kind in kinds:
            ck=torch.load(out/f'fold{fold}'/f'{kind}.pth',weights_only=False,map_location='cpu');te=ck['test']
            en=(RawProject() if kind=='raw' else Encoder64()).to(args.device);en.load_state_dict(ck['encoder']);en.eval()
            head=Head().to(args.device);head.load_state_dict(ck['classifier']);head.eval()
            with torch.no_grad():
                if kind=='raw':z=en(((x[te]-ck['raw_mean'])/ck['raw_std']).to(args.device))
                else:z=extract(en,ae_norm(x[te]).to(args.device))
                logits=head((z-ck['mu'].to(args.device))/ck['sd'].to(args.device));pred=logits.argmax(1).cpu().numpy()
            truth=y[te].numpy();report['folds'][str(fold)][kind]['test']=metric(truth,pred)
            save_json(out/f'fold{fold}'/f'{kind}_predictions.json',dict(true=truth.tolist(),pred=pred.tolist(),logits=logits.cpu().tolist(),confusion=confusion_matrix(truth,pred,labels=np.arange(24)).tolist()))
    report['summary']={}
    for kind in kinds+['validation_selected']:
        rows=[report['folds'][str(f)][report['selected'][str(f)] if kind=='validation_selected' else kind]['test'] for f in (1,2,3)]
        report['summary'][kind]=summarize(rows)
    save_json(out/'results.json',report);print(json.dumps(report['summary']),flush=True)

if __name__=='__main__':main()
