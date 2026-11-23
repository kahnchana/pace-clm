"""Check causal inputs and failure handling for the Sol answer generator."""
import pytest
from PIL import Image

from jevany_openai_answer import CausalAnswerRuntime, MODEL, SYSTEM, answer


def test_payload_preserves_causal_history_and_images():
    provider = CausalAnswerRuntime(model_name=MODEL, api_key='test-secret')
    image = Image.new('RGB', (384, 384))
    messages = [
        {'role':'system','content':[{'type':'text','text':'controller prompt'}]},
        {'role':'user','content':[{'type':'text','text':'Question and aged-out memory'}]},
        {'role':'assistant','content':[{'type':'text','text':'</Response> Prior answer'}]},
        {'role':'user','content':[{'type':'text','text':'Time 5.0s'}, {'type':'image','image':image}]},
    ]
    payload,audit = provider._payload(messages,max_new_tokens=96)
    assert payload['model']==MODEL
    assert payload['messages'][0]['content']==SYSTEM
    assert [m['role'] for m in payload['messages']]==['system','user','assistant','user']
    assert payload['messages'][1]['content'][0]['text']=='Question and aged-out memory'
    assert payload['messages'][2]['content'][0]['text']=='</Response> Prior answer'
    assert payload['messages'][3]['content'][1]['image_url']['url'].startswith('data:image/jpeg;base64,')
    assert payload['max_completion_tokens']==96
    assert payload['reasoning_effort']=='none'
    assert 'response_format' not in payload
    assert 'test-secret' not in str(audit)
    assert messages[0]['content'][0]['text']=='controller prompt'


@pytest.mark.parametrize('content,status',[('',200),('answer',503)])
def test_provider_failure_never_invents_an_answer(content,status):
    class Fake:
        last_call = {'status':status,'errors':[]}
        def generate_messages(self,messages,*,max_new_tokens):
            return content
    with pytest.raises(RuntimeError,match='Sol answer failed'):
        answer(Fake(),[],96)
