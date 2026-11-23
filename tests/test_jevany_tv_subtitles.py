from PIL import Image

from stream_gemma.episode import STATE_TAGS
from stream_gemma.inference.qwen_stream import stream_v25_qwen
from jevany_openai_answer import CausalAnswerRuntime, MODEL


def test_controller_and_sol_receive_only_current_causal_subtitle():
    class Runtime:
        frame_fps = 1
        def _resize(self, images):
            return images
        def generate_turn(self, messages, **kwargs):
            observed.append(messages)
            return STATE_TAGS['Silence']
    observed = []
    turns = [{'time':.2,'content':'Subtitle: first'}, {'time':1.4,'content':'Subtitle: second'},
             {'time':4.,'content':'Subtitle: future'}]
    stream_v25_qwen(Runtime(), [Image.new('RGB',(2,2)) for _ in range(3)],
        question='q', system_prompt='unchanged controller prompt',frame_times=[0.,1.,2.],
        extra_turns=turns,memory_mode='last-response',memory_instruction_version='v2')
    provider = CausalAnswerRuntime(model_name=MODEL,api_key='test')
    for index,messages in enumerate(observed):
        text = '\n'.join(p.get('text','') for m in messages for p in m['content'])
        assert 'Subtitle: future' not in text
        assert ('Subtitle: first' in text)==(index==1)
        assert ('Subtitle: second' in text)==(index==2)
        payload,_ = provider._payload(messages,max_new_tokens=96)
        api_text = str(payload['messages'])
        assert ('Subtitle: first' in api_text)==(index==1)
        assert ('Subtitle: second' in api_text)==(index==2)
        assert 'Subtitle: future' not in api_text
