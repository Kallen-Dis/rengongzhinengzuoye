"""训练后从原始预测独立复算指标，避免报告与输出不一致。"""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent

def main():
 r=ROOT/'results';s=json.loads((r/'summary.json').read_text(encoding='utf-8'))
 split=np.load(r/'split_indices.npz');tr=split['train'];va=split['validation']
 assert len(tr)==45000 and len(va)==5000 and len(np.unique(np.concatenate([tr,va])))==50000
 assert set(tr).isdisjoint(set(va))
 for m in s['models']:
  raw=np.load(r/f'{m["name"]}_predictions.npz');y=raw['truth'];p=raw['prediction'];probs=raw['probabilities']
  assert y.shape==p.shape==(10000,) and probs.shape==(10000,10)
  assert np.allclose(probs.sum(1),1,atol=1e-5) and np.all(p==probs.argmax(1))
  cm=np.zeros((10,10),dtype=int)
  for true,pred in zip(y,p):cm[true,pred]+=1
  assert np.array_equal(cm,np.array(m['confusion_matrix'])) and np.all(cm.sum(1)==1000)
  assert abs(float((y==p).mean())-m['test_accuracy'])<1e-9
  h=json.loads((r/f'{m["name"]}_history.json').read_text())
  assert len(h)==s['epochs'] and m['best_epoch']==max(h,key=lambda z:z['val_accuracy'])['epoch']
  assert np.all(np.isfinite([[v['train_loss'],v['val_loss']] for v in h]))
 assert s['selected_by_validation']==max(s['models'],key=lambda m:m['validation_accuracy'])['name']
 (r/'verification.json').write_text(json.dumps({'passed':True,'checks':['split_disjoint_and_complete','prediction_shape','probability_sum','independent_confusion_matrix','independent_accuracy','checkpoint_epoch_selection','finite_losses','model_selection_by_validation']},indent=2),encoding='utf-8')
 print('8 groups of result checks passed.')
if __name__=='__main__':main()
