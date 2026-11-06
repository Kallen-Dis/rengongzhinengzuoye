"""从真实训练日志生成曲线、混淆矩阵、误分类及 Grad-CAM。"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torchvision.datasets import CIFAR10
from train import Classifier,preprocess,CLASSES
ROOT=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(4)
    results=ROOT/'results'; out=results/'figures'; out.mkdir(exist_ok=True)
    summary=json.loads((results/'summary.json').read_text(encoding='utf-8'))
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    for s in summary['models']:
        h=json.loads((results/f'{s["name"]}_history.json').read_text())
        for field,style in [('train_accuracy','--'),('val_accuracy','-')]:
            axes[0].plot([x['epoch'] for x in h],[x[field]*100 for x in h],style,label=s['name']+' '+field.split('_')[0])
        for field,style in [('train_loss','--'),('val_loss','-')]:
            axes[1].plot([x['epoch'] for x in h],[x[field] for x in h],style,label=s['name']+' '+field.split('_')[0])
    for ax,title in zip(axes,['Accuracy (%)','Loss']):
        ax.set(xlabel='Epoch',ylabel=title); ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out/'curves.png',dpi=180); plt.close(fig)
    selected=summary['selected_by_validation']; s=next(s for s in summary['models'] if s['name']==selected)
    cm=np.array(s['confusion_matrix']); fig,ax=plt.subplots(figsize=(6.5,5.7))
    image=ax.imshow(cm,cmap='Blues'); fig.colorbar(image,ax=ax,label='Count')
    for i in range(10):
        for j in range(10): ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=7,color='white' if cm[i,j]>500 else 'black')
    ax.set(xticks=range(10),yticks=range(10),xticklabels=CLASSES,yticklabels=CLASSES,xlabel='Predicted class',ylabel='True class',title=f'{selected}: test confusion matrix')
    plt.setp(ax.get_xticklabels(),rotation=45,ha='right'); fig.tight_layout(); fig.savefig(out/'confusion.png',dpi=180); plt.close(fig)
    dataset=CIFAR10(str(ROOT/'data'),train=False,download=False)
    prediction=np.load(results/f'{selected}_predictions.npz')
    truth=prediction['truth']; pred=prediction['prediction']; probs=prediction['probabilities']
    # 按测试集索引取不同真实类别的错误，不按置信度挑选最戏剧化的样例。
    wrong=np.flatnonzero(truth!=pred); chosen=[]
    for c in range(10):
        candidates=wrong[truth[wrong]==c]
        if len(candidates): chosen.append(int(candidates[0]))
    fig,axes=plt.subplots(2,5,figsize=(10,5.8),layout='constrained')
    for ax,i in zip(axes.flat,chosen):
        ax.imshow(dataset.data[i]); ax.set_title(f'#{i} true:{CLASSES[truth[i]]}\npred:{CLASSES[pred[i]]} ({probs[i,pred[i]]:.2f})',fontsize=8); ax.axis('off')
    fig.savefig(out/'errors.png',dpi=180); plt.close(fig)
    (results/'error_examples.json').write_text(json.dumps([dict(index=i,true=CLASSES[truth[i]],predicted=CLASSES[pred[i]],confidence=float(probs[i,pred[i]])) for i in chosen],indent=2),encoding='utf-8')
    model=Classifier(selected=='improved'); model.load_state_dict(torch.load(ROOT/'checkpoints'/f'{selected}.pt',map_location='cpu',weights_only=True)['model']); model.eval()
    activations=[]
    def capture(module,inputs,output): output.retain_grad(); activations.append(output)
    handle=model.features[-1].register_forward_hook(capture)
    fig,axes=plt.subplots(2,4,figsize=(8,4.3))
    for col,i in enumerate(chosen[:4]):
        model.zero_grad(); activations.clear()
        tensor=torch.tensor(dataset.data[i].transpose(2,0,1).copy()).unsqueeze(0)
        logits=model(preprocess(tensor,False)); logits[0,pred[i]].backward()
        feature=activations[0]; weights=feature.grad.mean((2,3),keepdim=True)
        cam=(feature*weights).sum(1,keepdim=True).relu()
        cam=torch.nn.functional.interpolate(cam,size=(32,32),mode='bilinear',align_corners=False)[0,0].detach().numpy()
        cam=(cam-cam.min())/(cam.max()-cam.min()+1e-8)
        axes[0,col].imshow(dataset.data[i]); axes[1,col].imshow(dataset.data[i]); axes[1,col].imshow(cam,cmap='jet',alpha=.45,vmin=0,vmax=1)
        axes[0,col].set_title(f'#{i}: {CLASSES[pred[i]]}',fontsize=9)
        axes[0,col].axis('off'); axes[1,col].axis('off')
    handle.remove(); fig.tight_layout(); fig.savefig(out/'gradcam.png',dpi=180); plt.close(fig)
    assert cm.sum()==10000 and np.all(cm.sum(1)==1000)
    assert np.isclose(np.trace(cm)/10000,s['test_accuracy'])
    print('All figures and metric consistency checks completed.')
if __name__=='__main__': main()
