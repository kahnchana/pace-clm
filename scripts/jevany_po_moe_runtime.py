"""The committed 2B argmax controller and frozen promoted MoE for SB streams."""
import time
import torch

from generate_selected_responses import forced_response
from pace_clm.jevany_streaming import ACTIONS, PointerHead, controller_logits, decision_inputs
from pace_clm.qwen_controller_runtime import load_dense_controller
from pace_clm.stream_gemma_adapter import validate_promoted_contract
from stream_gemma.episode import STATE_TAGS
from stream_gemma.inference.qwen_stream import QwenRuntime
from eval_jevany_tv_vad160 import SG, read
from eval_jevany_streaming import with_options


class ControllerMoE:
    def __init__(self,checkpoint):
        identity = read(checkpoint/'model_identity.json')
        self.runtime,_ = load_dense_controller(model_name=identity['model_name'],
            model_revision=identity['model_revision'],checkpoint=str(checkpoint/'controller_adapter'))
        contract = validate_promoted_contract(SG)
        self.generator = QwenRuntime.load(checkpoint=contract['adapter'],model_name=contract['model_name'],
            model_revision=contract['model_revision'],frame_num_tokens=144,generation_prefix_mode='direct')
        self.head = PointerHead(int(self.runtime.model.config.text_config.hidden_size)).to(self.runtime.device)
        self.head.load_state_dict(torch.load(checkpoint/'pointer_head.pt',map_location='cpu',weights_only=True))
        self.head.eval();self.runtime.model.eval();self.generator.model.eval()
        self.runtime.generate_turn = self.choose_turn
        self.reset()

    def reset(self):
        self.trace = []
        self.generator_frames = {}

    def choose_turn(self,messages,*,max_new_tokens,vision=None):
        if vision is None:
            raise RuntimeError('controller vision cache required')
        started = time.monotonic()
        inputs,options,decision = decision_inputs(self.runtime,with_options(messages),vision)
        with torch.inference_mode():
            logits = controller_logits(self.runtime,self.head,inputs,vision,options,decision,train_decoder=False)
        if not torch.isfinite(logits).all():
            raise ValueError('nonfinite learned action scores')
        action = ACTIONS[int(logits.argmax().item())]
        row = dict(action=action,logits=logits.detach().float().cpu().tolist(),
                   observation=messages[-1]['content'][0].get('text',''),input_frames=len(vision),
                   input_tokens=int(inputs['input_ids'].shape[1]),decision_seconds=time.monotonic()-started)
        self.trace.append(row)
        if action!='Response':
            return STATE_TAGS['Silence' if action=='Silence' else 'Standby']
        images = [part['image'] for message in messages for part in message['content'] if part.get('type')=='image']
        missing = [image for image in images if id(image) not in self.generator_frames]
        if missing:
            frames = [self.generator.encode_frame(image) for image in self.generator._resize(missing)]
            self.generator.encode_frames_features(frames)
            self.generator_frames.update((id(image),(image,frame)) for image,frame in zip(missing,frames,strict=True))
        generator_vision = [self.generator_frames[id(image)][1] for image in images]
        started = time.monotonic()
        content = forced_response(self.generator,messages,generator_vision,min(128,max_new_tokens))
        row['generation_seconds'] = time.monotonic()-started
        if not content.strip():
            raise ValueError('empty MoE response')
        return f"{STATE_TAGS['Response']} {content}"
