# PAUC-F–aligned RLOO for proactive streaming

We train a streaming controller with **REINFORCE Leave-One-Out (RLOO)** to
maximize the PAUC-F score of its complete response sequence. The controller
learns when to respond, remain silent, or hold a response as evidence arrives.
A frozen answer generator supplies the text whenever the controller responds;
RL updates the controller's LoRA parameters and action head.

PAUC-F is the task reward, rather than a differentiable loss. The current
implementation also uses training-only regularization and a supervised anchor,
so describing the total loss as “entirely PAUC-F” would be inaccurate.

## Causal streaming policy

At decision step $t$, the controller observes the current visual evidence, the
user request, recent history, and memory of earlier responses. It cannot inspect
future evidence. Its three actions are `Silence`, `Standby` (hold), and `Response`.
Each sampled action and any generated answer become part of the history used
for subsequent decisions. Trajectories therefore reflect the consequences of
the controller's own earlier choices.

```python
# High-level pseudocode; action order matches the runtime.
SILENCE, STANDBY, RESPONSE = range(3)

def rollout(controller, frozen_generator, stream, request):
    history, decisions, responses = [], [], []
    for timestamp, evidence in stream:
        observation = causal_observation(request, evidence, history)
        logits = controller(observation)
        action = Categorical(logits=logits).sample()
        answer = frozen_generator(observation) if action == RESPONSE else ""
        decisions.append((observation, action))
        if action == RESPONSE:
            responses.append((timestamp, answer))
        history.append((timestamp, action, answer))
    return {"observations": [o for o, a in decisions],
            "actions": torch.tensor([a for o, a in decisions]),
            "responses": responses}
```

## Episode reward and RLOO credit

PAUC-F combines a time-sensitive, judged answer-quality recall component $R$
with response precision $P$:

$$
P = \frac{\text{value-adding responses}}{\text{all emitted responses}},
\qquad
r = \frac{\mathrm{PAUC\text{-}F}}{100}
  = \frac{2PR}{P+R}.
$$

The score is zero when the denominator is zero. Precision credits at most one
response per target turn, using its earliest positively judged response;
additional emissions still count in the denominator. This rewards timely,
useful coverage while discouraging redundant or unhelpful interruptions.
The scorer uses $\omega=0.5$ for the recall component. For an episode with no
target response, the implementation instead assigns $-0.25$ per emission.

For each episode, we sample $K=4$ complete trajectories from the same current
policy. Each trajectory receives its terminal reward. RLOO compares that reward
with the mean reward of the other trajectories from the same episode:

$$
A_k = r_k - \frac{1}{K-1}\sum_{j\ne k} r_j,
\qquad
\mathcal{L}_{\mathrm{RLOO}}
= -\frac{1}{C}\sum_k A_k\sum_t
\log\pi_\theta(a_{k,t}\mid o_{k,t}).
$$

The same detached advantage weights every decision in a trajectory, including
silence and hold actions. This is episode-level credit assignment, without a
learned critic or per-frame reward model. RLOO subtracts the peer mean without
dividing by a group standard deviation. If all four rewards match, the group's
RLOO gradient is zero.

```python
# Schematic core update; production also applies batch weights and regularizers.
traces = [rollout(policy, generator, stream, request) for _ in range(4)]
rewards = torch.tensor([score_episode(t["responses"]) for t in traces])
baselines = (rewards.sum() - rewards) / (len(rewards) - 1)
advantages = (rewards - baselines).detach()

loss = 0.0
for trace, advantage in zip(traces, advantages):
    logits = policy(trace["observations"])  # Recompute with gradients enabled.
    logp = logits.log_softmax(dim=-1)
    chosen_logp = logp.gather(-1, trace["actions"][:, None]).squeeze(-1)
    loss -= advantage * chosen_logp.sum()
loss /= len(traces) * 32
```

The examples show control flow rather than runnable APIs. Production sums over
complete trajectories and normalizes globally by the trajectory count times a
fixed scale of 32, with per-trajectory weights. It does not divide each
trajectory's contribution by its length. Rollouts are on-policy, and replayed
action distributions are checked against collection before updating.

## Full optimization objective

The implemented decision loss is
`-advantage * log_prob + 0.02 * KL(policy || reference) - 0.001 * entropy`,
with the weighting and normalization above. A fixed-weight supervised
cross-entropy anchor is added to retain established action behavior. The
reference policy and answer generator stay frozen. No gradient passes through
the judge, PAUC-F calculation, or generated answer text.

At inference, the controller selects `logits.argmax(-1)` directly. Learned
weights determine response behavior; there are no score offsets, response
thresholds, temperature calibration, or cooldowns. The compatibility response
margin is zero.

## Contribution for proactive streaming

The contribution is applying **whole-episode, metric-aligned policy optimization
to causal response control**. A proactive system must decide both when an
answer is warranted and when another interruption adds no value. Optimizing the
quality of an isolated answer does not capture those coupled decisions.
PAUC-F evaluates their combined outcome, and RLOO turns differences between
complete response schedules into updates to the streaming controller.

Because earlier responses change later context, this learns a sequential
interaction policy rather than independent frame classifications. Keeping the
generator fixed isolates improvements to learned response control. RLOO and
PAUC-F themselves are existing ingredients; the methodological contribution is
their integration for this streaming objective, rather than a claim of a new
RL estimator or a verified first use.

Implementation references: [RLOO advantages](../src/pace_clm/rl.py),
[weighted decision loss](../src/pace_clm/rl_mixed.py),
[episode scoring](../scripts/jevany_mixed_reward.py),
[rollout and learner](../scripts/jevany_rl_mixed_workers.py), and
[supervised anchor](../src/pace_clm/rl_anchor.py).
