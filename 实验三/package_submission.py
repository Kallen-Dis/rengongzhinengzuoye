"""打包可复现实验材料；原始数据不重复打包，使用README中的下载入口。"""
from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
 files=list(ROOT.glob('*.py'))+[ROOT/'README.md',ROOT/'requirements.txt',ROOT/'.gitignore']
 for folder in ['results','checkpoints','report']:
  files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file())
 assert (ROOT/'results/verification.json').is_file() and (ROOT/'report/实验三报告.pdf').is_file()
 archive=ROOT/'实验三提交材料.zip'
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
  for p in sorted(files):z.write(p,'实验三/'+p.relative_to(ROOT).as_posix())
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None
  assert len([n for n in z.namelist() if n.endswith('.pdf')])==1
 print(archive,'files',len(files),'bytes',archive.stat().st_size)
