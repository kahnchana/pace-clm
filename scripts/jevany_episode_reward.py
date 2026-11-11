"""Episode reward using the same official scorer and complete-request GPT cache."""
from jevany_rl_reward import contract as clip_contract, score_traces as official_score, write

VERSION='episode-pauc-f-gpt-v1'


def contract():
    return {**clip_contract(), 'version':VERSION,
            'event_reward':'official whole-episode PAUC-F / 100',
            'initialization':'empty history at frame zero',
            'quiet_sections':'all physical fires in episode precision denominator',
            'aggregation_limit':'mean episode return is not benchmark-global PAUC-F',
            'gpt_cache_identity':'unchanged complete request/model/rubric identity'}


def score_traces(traces,directory,cache,**kwargs):
    report=official_score(traces,directory,cache,**kwargs)
    for trace in traces:trace['reward_version']=VERSION
    write(directory/'contract.json',contract())
    return report
