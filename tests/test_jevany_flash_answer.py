"""Flash must retain causal history, current subtitles, and fail on API errors."""
import pytest
from PIL import Image
from jevany_gemini_answer import CausalFlashRuntime, MODEL, SYSTEM, answer


def test_causal_history_subtitles_and_images():
    provider = CausalFlashRuntime(model_name=MODEL, project='test')
    messages = [
        {'role':'system','content':[{'type':'text','text':'controller prompt'}]},
        {'role':'user','content':[{'type':'text','text':'Request and aged-out memory'}]},
        {'role':'assistant','content':[{'type':'text','text':'</Response> Prior answer'}]},
        {'role':'user','content':[{'type':'text','text':'Time 5s. Subtitle: current evidence'},
                                  {'type':'image','image':Image.new('RGB',(384,384))}]},
    ]
    payload, audit = provider._payload(messages,max_new_tokens=96)
    assert payload['systemInstruction']=={'parts':[{'text':SYSTEM}]}
    assert [m['role'] for m in payload['contents']]==['user','model','user']
    assert payload['contents'][1]['parts']==[{'text':'</Response> Prior answer'}]
    assert payload['contents'][2]['parts'][0]['text']=='Time 5s. Subtitle: current evidence'
    assert payload['contents'][2]['parts'][1]['inlineData']['mimeType']=='image/jpeg'
    assert payload['generationConfig']['maxOutputTokens']==224
    assert audit['visible_output_token_budget']==96
    assert messages[0]['content'][0]['text']=='controller prompt'


@pytest.mark.parametrize('content,status',[('',200),('answer',503)])
def test_failed_calls_never_create_fallback_answers(content,status):
    class Failed:
        last_call={'status':status,'errors':[]}
        def generate_messages(self,messages,*,max_new_tokens): return content
    with pytest.raises(RuntimeError,match='Flash answer failed'):
        answer(Failed(),[],96)
