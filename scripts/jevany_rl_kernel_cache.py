#!/usr/bin/env python3
"""Persist content-addressed Triton binaries across Slurm's temporary directories."""
import argparse,hashlib,json,os,shutil,time,uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path


def main():
 p=argparse.ArgumentParser()
 p.add_argument('mode',choices=('stage','save'))
 p.add_argument('--local',type=Path,default=Path('/tmp/pace-rl-triton'))
 p.add_argument('--shared',type=Path,default=Path('outputs/jevany-2b-rl/kernel-cache'))
 p.add_argument('--audit',type=Path)
 a=p.parse_args();started=time.monotonic()
 source,dest=(a.shared,a.local) if a.mode=='stage' else (a.local,a.shared)
 files=[f for f in source.rglob('*') if f.is_file() and not f.is_symlink()
        and '.tmp' not in f.name and f.name!='lock' and not f.name.endswith('.lock')] if source.exists() else []
 def copy_file(path):
  target=dest/path.relative_to(source)
  if target.exists():return 0
  target.parent.mkdir(parents=True,exist_ok=True)
  temporary=target.with_name(target.name+'.tmp.'+str(os.getpid())+'.'+uuid.uuid4().hex)
  try:
   shutil.copy2(path,temporary)
   size=temporary.stat().st_size
   os.replace(temporary,target)
   return size
  except FileNotFoundError:
   temporary.unlink(missing_ok=True)
   return 0
 with ThreadPoolExecutor(max_workers=8) as pool:copied=sum(pool.map(copy_file,files))
 result={'mode':a.mode,'utc':datetime.now(timezone.utc).isoformat(),'files_seen':len(files),
         'bytes_copied':copied,'seconds':time.monotonic()-started,
         'source':str(source),'destination':str(dest),
         'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 if a.audit:
  a.audit.parent.mkdir(parents=True,exist_ok=True)
  with a.audit.open('a') as handle:handle.write(json.dumps(result)+'\n')
 print(json.dumps(result),flush=True)


if __name__=='__main__':main()
