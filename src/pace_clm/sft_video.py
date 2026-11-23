"""Bounded persistent video decoders; exact requested physical frame indices."""
from collections import OrderedDict,defaultdict


class SequentialVideoCache:
    def __init__(self,maximum=32):
        self.maximum=maximum;self.entries=OrderedDict()

    def decode(self,path,indices):
        import cv2
        from PIL import Image
        if path not in self.entries:
            cap=cv2.VideoCapture(path,cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])
            if not cap.isOpened():raise RuntimeError('cannot open '+path)
            self.entries[path]=[cap,0]
        self.entries.move_to_end(path)
        cap,position=self.entries[path];images=[]
        for index in indices:
            if index<position or index-position>32:
                if not cap.set(cv2.CAP_PROP_POS_FRAMES,index):raise RuntimeError('seek failed')
                position=index
            while position<index:
                if not cap.grab():raise RuntimeError('frame skip failed')
                position+=1
            ok,frame=cap.read()
            if not ok:raise RuntimeError(f'cannot decode {path} frame {index}')
            images.append(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)))
            position=index+1
        self.entries[path][1]=position
        while len(self.entries)>self.maximum:
            _,(old,_)=self.entries.popitem(last=False);old.release()
        return images

    def prepare_rows(self,runtime,rows,cached):
        from pace_clm.rl_mixed import physical_row
        requested=defaultdict(set);result=dict(cached)
        for row in rows:
            physical=physical_row(row);path=physical['video_path']
            requested[path].update(t['frame_index'] for t in physical['turns'] if (path,t['frame_index']) not in result)
        for path,indices in requested.items():
            ordered=sorted(indices)
            images=runtime._resize(self.decode(path,ordered))
            result.update(((path,i),image) for i,image in zip(ordered,images,strict=True))
        return result

    def close(self):
        for cap,_ in self.entries.values():cap.release()
        self.entries.clear()
