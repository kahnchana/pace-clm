"""Causal prose answers through the same authenticated gateway as GPT scoring."""
from stream_gemma.inference.closed_api import OpenAICompatibleRuntime

MODEL = 'gpt-5.6-sol'
SYSTEM = (
    "Answer the standing video request using only frames observed through the "
    "current timestamp and the supplied history. Give one concise factual "
    "sentence in plain text. Prior streaming action tags are history; the "
    "controller has selected a response now. Do not emit streaming control tags."
)


class CausalAnswerRuntime(OpenAICompatibleRuntime):
    """Reuse image encoding, authentication and retries without ZSS JSON prompts."""

    def _payload(self, messages, *, max_new_tokens):
        provider_messages = []
        audit_parts = []
        for message in messages:
            role = message['role']
            if role == 'system':
                provider_messages.append(dict(role='system', content=SYSTEM))
                continue
            content = []
            for part in message['content']:
                if part['type'] == 'text':
                    value = dict(type='text', text=part['text'])
                    content.append(value)
                    audit_parts.append(value)
                elif part['type'] == 'image':
                    value, audit = self._encode_image(part['image'])
                    content.append(value)
                    audit_parts.append(audit)
                else:
                    raise ValueError('unsupported causal input part')
            provider_messages.append(dict(role=role, content=content))
        config = dict(max_completion_tokens=int(max_new_tokens), reasoning_effort='none')
        return dict(model=self.model_name, messages=provider_messages, **config), dict(
            provider='salesforce-openai-gateway', model=self.model_name, base_url=self.base_url,
            system_instruction=SYSTEM, user_parts=audit_parts, generation_config=config)


def load_provider():
    from judge_exp10_utility import load_gateway_key
    load_gateway_key()
    return CausalAnswerRuntime.load(model_name=MODEL, max_attempts=5, timeout=120)


def answer(provider, messages, max_new_tokens):
    content = provider.generate_messages(messages, max_new_tokens=max_new_tokens)
    call = provider.last_call
    if call.get('status') != 200 or not content.strip():
        raise RuntimeError(f"Sol answer failed: status={call.get('status')}, errors={call.get('errors')}")
    return content
