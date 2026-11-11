#!/usr/bin/env python3
"""Record content hashes of every frozen training video without loading a GPU."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path


def checksum(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    paths=sorted({Path(json.loads(line)['video_path']) for line in args.manifest.open()})
    def one(path):
        before=path.stat()
        digest=checksum(path)
        after=path.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
            raise ValueError('video mutated while hashing: '+str(path))
        return {'path':str(path),'size_bytes':after.st_size,'mtime_ns':after.st_mtime_ns,'sha256':digest}
    temporary=args.output.with_suffix('.tmp')
    total=0
    with temporary.open('w') as f, ThreadPoolExecutor(max_workers=8) as pool:
        for count,row in enumerate(pool.map(one,paths),1):
            f.write(json.dumps(row,sort_keys=True)+'\n');total+=row['size_bytes']
            if count%500==0:print(json.dumps({'hashed_videos':count,'total':len(paths),'bytes':total}),flush=True)
    temporary.replace(args.output)
    summary={'manifest_sha256':checksum(args.manifest),'videos_sha256':checksum(args.output),
             'script_sha256':checksum(Path(__file__)),'videos':len(paths),'bytes':total,
             'split':'train; frozen episode manifest exclusions apply'}
    args.output.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
