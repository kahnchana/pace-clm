"""Flash prose answers preserving the Sol causal message content and roles."""
from stream_gemma.inference.closed_api import VertexGeminiRuntime
from jevany_openai_answer import SYSTEM
MODEL = 'gemini-3.8-flash'

class CausalFlashRuntime(VertexGeminiRuntime):
    def _payload(self, messages, *, max_new_tokens):
        payload, audit = super()._payload(messages, max_new_tokens=max_new_tokens)
        contents = []
        for message in messages:
            if message['role'] == 'system':
                continue
            parts = []
            for part in message['content']:
                if part['type'] == 'text':
                    parts.append({'text': part['text']})
                elif part['type'] == 'image':
                    encoded, _ = self._encode_image(part['image'])
                    parts.append(encoded)
                else:
                    raise ValueError('unsupported causal input part')
            contents.append({'role': 'model' if message['role'] == 'assistant' else 'user', 'parts': parts})
        payload['contents'] = contents
        payload['systemInstruction'] = {'parts': [{'text': SYSTEM}]}
        audit['system_instruction'] = SYSTEM
        audit['message_roles'] = [m['role'] for m in contents]
        return payload, audit

def load_provider():
    return CausalFlashRuntime.load(model_name=MODEL, location='global', max_attempts=5, timeout=120)

def answer(provider, messages, max_new_tokens):
    content = provider.generate_messages(messages, max_new_tokens=max_new_tokens)
    call = provider.last_call
    if call.get('status') != 200 or not content.strip():
        raise RuntimeError(f"Flash answer failed: status={call.get('status')}, errors={call.get('errors')}")
    return content
