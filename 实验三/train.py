"""CIFAR-10：固定分层划分、从零训练两种CNN、验证集选模型、最后测试。"""
import argparse, csv, hashlib, json, random, time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import CIFAR10

ROOT=Path(__file__).resolve().parent
CLASSES=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

class SE(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.net=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(c,max(c//8,4),1),nn.ReLU(),nn.Conv2d(max(c//8,4),c,1),nn.Sigmoid())
    def forward(self,x): return x*self.net(x)

class Block(nn.Module):
    def __init__(self,a,b,residual=False):
        super().__init__(); self.residual=residual
        self.convs=nn.Sequential(nn.Conv2d(a,b,3,padding=1,bias=False),nn.BatchNorm2d(b),nn.ReLU(),nn.Conv2d(b,b,3,padding=1,bias=False),nn.BatchNorm2d(b))
        self.shortcut=nn.Conv2d(a,b,1,bias=False) if residual else nn.Identity()
        self.se=SE(b) if residual else nn.Identity()
    def forward(self,x):
        z=self.se(self.convs(x))
        if self.residual: z=z+self.shortcut(x)
        return F.max_pool2d(F.relu(z),2)

class Classifier(nn.Module):
    def __init__(self,improved=False):
        super().__init__()
        self.features=nn.Sequential(Block(3,32,improved),Block(32,64,improved),Block(64,128,improved))
        self.head=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Dropout(.2),nn.Linear(128,10))
    def forward(self,x): return self.head(self.features(x))

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def prepare():
    archive=ROOT/'data/cifar-10-python.tar.gz'
    if archive.exists():
        assert hashlib.md5(archive.read_bytes()).hexdigest()=='c58f30108f718f92721af3b95e74349a','数据归档校验失败'
    train=CIFAR10(str(ROOT/'data'),train=True,download=True)
    test=CIFAR10(str(ROOT/'data'),train=False,download=True)
    rng=np.random.default_rng(2026); labels=np.array(train.targets)
    tr=[]; va=[]
    for c in range(10):
        ids=rng.permutation(np.flatnonzero(labels==c)); va.extend(ids[:500]); tr.extend(ids[500:])
    tr=np.array(tr); va=np.array(va)
    assert len(set(tr)&set(va))==0 and len(tr)==45000 and len(va)==5000
    (ROOT/'results').mkdir(exist_ok=True)
    np.savez(ROOT/'results/split_indices.npz',train=tr,validation=va)
    return train,test,tr,va

def preprocess(x,augment):
    x=x.float()/255
    if augment:
        x=F.pad(x,(4,4,4,4),mode='reflect')
        b=x.shape[0]; shifts=torch.randint(0,9,(b,2),device=x.device)
        rows=shifts[:,0,None,None]+torch.arange(32,device=x.device)[None,:,None]
        cols=shifts[:,1,None,None]+torch.arange(32,device=x.device)[None,None,:]
        x=x.permute(0,2,3,1)[torch.arange(b,device=x.device)[:,None,None],rows,cols].permute(0,3,1,2)
        x=torch.where(torch.rand(b,1,1,1,device=x.device)<.5,x.flip(-1),x)
    # 固定映射至[-1,1]，不使用验证集或测试集统计量。
    return (x-.5)/.5

@torch.no_grad()
def evaluate(model,x,y,batch=256):
    model.eval(); losses=[]; preds=[]; probs=[]
    for start in range(0,len(y),batch):
        logits=model(preprocess(x[start:start+batch],False))
        losses.append(F.cross_entropy(logits,y[start:start+batch],reduction='sum').item())
        prob=logits.softmax(1); preds.append(prob.argmax(1).cpu()); probs.append(prob.cpu())
    pred=torch.cat(preds).numpy(); prob=torch.cat(probs).numpy()
    return sum(losses)/len(y),float((pred==y.cpu().numpy()).mean()),pred,prob

def run(args):
    started_at=datetime.now(timezone(timedelta(hours=8))).isoformat()
    seed_all(2026); torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark=False
    train,test,tr,va=prepare()
    x=torch.tensor(train.data.transpose(0,3,1,2),device=device)
    y=torch.tensor(train.targets,device=device)
    xt=x[torch.tensor(tr,device=device)]; yt=y[torch.tensor(tr,device=device)]
    xv=x[torch.tensor(va,device=device)]; yv=y[torch.tensor(va,device=device)]
    del x,y
    results=ROOT/'results'; checkpoints=ROOT/'checkpoints'; checkpoints.mkdir(exist_ok=True)
    summaries=[]
    for name in args.models:
        seed_all(2026); improved=name=='improved'
        model=Classifier(improved).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0005)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,eta_min=.00002)
        scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
        history=[]; best=-1.; start_time=time.perf_counter()
        for epoch in range(1,args.epochs+1):
            model.train(); order=torch.randperm(len(yt),device=device)
            total_loss=torch.zeros((),device=device); correct=torch.zeros((),device=device)
            lr=optimizer.param_groups[0]['lr']; tick=time.perf_counter()
            for ids in order.split(args.batch):
                bx=preprocess(xt[ids],improved); by=yt[ids]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
                    logits=model(bx); loss=F.cross_entropy(logits,by,label_smoothing=.1 if improved else 0.)
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                total_loss+=loss.detach()*len(ids); correct+=(logits.argmax(1)==by).sum()
            vl,acc,_,_=evaluate(model,xv,yv,args.batch)
            row=dict(epoch=epoch,train_loss=total_loss.item()/len(yt),train_accuracy=correct.item()/len(yt),val_loss=vl,val_accuracy=acc,lr=lr,seconds=time.perf_counter()-tick)
            history.append(row); scheduler.step()
            if acc>best:
                best=acc; best_epoch=epoch
                torch.save({'model':model.state_dict(),'name':name,'epoch':epoch,'val_accuracy':acc},checkpoints/f'{name}.pt')
            (results/f'{name}_history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
            print(f'{name} {epoch}/{args.epochs} train={row["train_accuracy"]:.4f} val={acc:.4f} loss={vl:.4f} seconds={row["seconds"]:.1f}',flush=True)
        summaries.append(dict(name=name,best_epoch=best_epoch,validation_accuracy=best,parameters=sum(p.numel() for p in model.parameters()),training_seconds=time.perf_counter()-start_time))
    # 训练及验证选择全部结束后才运行测试评估；禁止用测试指标重新调参。
    testx=torch.tensor(test.data.transpose(0,3,1,2),device=device); testy=torch.tensor(test.targets,device=device)
    for s in summaries:
        model=Classifier(s['name']=='improved').to(device)
        model.load_state_dict(torch.load(checkpoints/f'{s["name"]}.pt',map_location=device,weights_only=True)['model'])
        loss,acc,pred,prob=evaluate(model,testx,testy,args.batch)
        cm=np.zeros((10,10),dtype=int); np.add.at(cm,(test.targets,pred),1)
        s.update(test_loss=loss,test_accuracy=acc,confusion_matrix=cm.tolist(),per_class_accuracy=(cm.diagonal()/cm.sum(1)).tolist())
        np.savez_compressed(results/f'{s["name"]}_predictions.npz',truth=test.targets,prediction=pred,probabilities=prob)
        print(s['name'],'TEST',acc,flush=True)
    output=dict(started_at=started_at,finished_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),seed=2026,epochs=args.epochs,batch=args.batch,train_size=45000,val_size=5000,test_size=10000,classes=CLASSES,device=str(device),gpu=torch.cuda.get_device_name(0) if device.type=='cuda' else None,torch=torch.__version__,selected_by_validation=max(summaries,key=lambda s:s['validation_accuracy'])['name'],models=summaries)
    (results/'summary.json').write_text(json.dumps(output,indent=2,ensure_ascii=False),encoding='utf-8')
    with (results/'per_class_accuracy.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f); w.writerow(['class']+[s['name'] for s in summaries])
        for i,c in enumerate(CLASSES): w.writerow([c]+[s['per_class_accuracy'][i] for s in summaries])

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--epochs',type=int,default=30); parser.add_argument('--batch',type=int,default=256)
    parser.add_argument('--models',nargs='+',choices=['baseline','improved'],default=['baseline','improved'])
    run(parser.parse_args())
