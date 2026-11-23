"""Capture a failing rank-7 request without changing model inputs or settings."""
import json
from pathlib import Path
import sys
import jevany_gemini_answer as provider_module
original=provider_module.answer
ROOT=Path(__file__).resolve().parents[1]
DIRECTORY=ROOT/'outputs/jevany-2b-rl/mixed-web-ego-estp-balanced-v4-to160/evaluation/web-ego-tv-subs-vad-flash-step160/web'

def audited_answer(provider,messages,max_new_tokens):
    try: return original(provider,messages,max_new_tokens)
    except Exception:
        payload,audit=provider._payload(messages,max_new_tokens=max_new_tokens)
        (DIRECTORY/'failed-flash-request.json').write_text(json.dumps(dict(payload=payload,audit=audit,endpoint=provider.endpoint)))
        (DIRECTORY/'failed-flash-response.json').write_text(json.dumps(provider.last_call,indent=2)+'\n')
        print('FAILED_PROVIDER_RESPONSE',json.dumps(provider.last_call.get('response')),flush=True)
        raise

provider_module.answer=audited_answer
import eval_jevany_flash_streaming as evaluator
sys.argv=[__file__,'--controller',str(DIRECTORY/'full_run/checkpoints/step-000160'),
          '--validation-manifest',str(DIRECTORY/'evaluation-manifest.jsonl'),
          '--selection-ids',str(DIRECTORY/'selection.json'),'--output-dir',str(DIRECTORY/'rollout'),
          '--response-margin','0','--answer-model','gemini-3.8-flash','--subtitles','off',
          '--max-new-tokens','96','--record-action-logits']
evaluator.main()
