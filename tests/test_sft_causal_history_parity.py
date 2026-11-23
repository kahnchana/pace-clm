"""Check the batched evaluator's prompt construction against the native harness."""
from pace_clm.jevany_streaming import decision_messages,option_text
from pace_clm.rl_mixed import observation
from stream_gemma.inference.qwen_stream import stream_v25_qwen
from stream_gemma.config import SYSTEM_PROMPT
from stream_gemma.v24_memory import normalize_memory_content


def test_native_harness_messages_match_through_memory_eviction():
    images=[f'image-{i}' for i in range(42)]
    spec=dict(span_id='test',video_id='video',video_path='video.mp4',question='Report new events',
        decode_start=0,source_indices=list(range(100,184,2)),timestamps=[i+.5 for i in range(42)])
    history=[];checked=[]

    class Stub:
        def _resize(self,values):return values

        def generate_turn(self,messages,**kwargs):
            index=len(checked)
            row=observation(spec,history,index)
            expected=decision_messages(row,images[max(0,index-31):index+1])
            expected[-1]['content']=[p for p in expected[-1]['content'] if p.get('text')!=option_text()]
            assert messages==expected
            if index in (3,37):emit='</Response> The event is visible.';state='Response'
            elif index in (4,5):emit='</Standby>';state='Standby'
            else:emit='</Silence>';state='Silence'
            turn=dict(frame_index=index,ts=row['turns'][-1]['ts'],emit=emit,state=state)
            if state=='Response':turn['memory_content']=normalize_memory_content('The event is visible.')
            history.append(turn);checked.append(index)
            return emit

    stream_v25_qwen(Stub(),images,question=spec['question'],system_prompt=SYSTEM_PROMPT,
        frame_times=spec['timestamps'],memory_mode='last-response',memory_instruction_version='v2',memory_select='aged-out')
    assert len(checked)==42
