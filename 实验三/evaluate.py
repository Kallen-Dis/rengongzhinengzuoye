"""直接加载最佳checkpoint复算测试准确率，无需重新训练。"""
import json
from pathlib import Path
import torch
from torchvision.datasets import CIFAR10
from train import Classifier,evaluate
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
 torch.set_num_threads(4)
 summary=json.loads((ROOT/'results/summary.json').read_text(encoding='utf-8'))
 dataset=CIFAR10(str(ROOT/'data'),train=False,download=True)
 device='cuda' if torch.cuda.is_available() else 'cpu'
 x=torch.tensor(dataset.data.transpose(0,3,1,2),device=device)
 y=torch.tensor(dataset.targets,device=device)
 checks=[]
 for row in summary['models']:
  model=Classifier(row['name']=='improved').to(device)
  checkpoint=torch.load(ROOT/'checkpoints'/f'{row["name"]}.pt',map_location=device,weights_only=True)
  model.load_state_dict(checkpoint['model'])
  loss,accuracy,_,_=evaluate(model,x,y)
  checks.append(dict(model=row['name'],accuracy=accuracy,reported_accuracy=row['test_accuracy'],absolute_difference=abs(accuracy-row['test_accuracy'])))
  print(row['name'],accuracy,flush=True)
 (ROOT/'results/checkpoint_evaluation.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
