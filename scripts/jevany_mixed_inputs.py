"""Decode native physical frames while retaining logical causal history slots."""
from train_jevany_streaming import EpisodeFrames as OriginalFrames
from benchmark_qwen35_2b_training import predecode_batch as original_predecode
from pace_clm.rl_mixed import physical_row


class EpisodeFrames(OriginalFrames):
    def prepare(self,row,*args,**kwargs):
        return super().prepare(physical_row(row),*args,**kwargs)

    def images_for(self,row):
        physical=physical_row(row)
        return [self.frames[(row['video_path'],t['frame_index'])][0] for t in physical['turns']]

    def features_for(self,row):
        physical=physical_row(row)
        return [self.frames[(row['video_path'],t['frame_index'])][1] for t in physical['turns']]


def predecode_batch(runtime,rows,pool,cached_images=None):
    return original_predecode(runtime,[physical_row(row) for row in rows],pool,cached_images)
