# Five generation Alpamayo result

The campaign finished five gameplay policy gradient updates, 40 training rollouts,
and 24 fixed evaluations. Each update changed the expert adapter weights and
passed save and reload verification. No supervised imitation was performed.

The five generation result does not establish useful driving improvement.
Standing starts remain stalled. No evaluation completed a lap. Moving evaluations
started at 5 m/s; simulator supplied setup motion was excluded from rewards.
Each evaluation allowed up to 30 seconds of model control, with early stall stops.

| Policy generation | Mean moving progress | Mean moving reward | Mean standing progress |
| --- | ---: | ---: | ---: |
| Baseline | 55.27 m | 0.5527 | 0.341 m |
| 1 | 36.28 m | 0.3628 | 0.339 m |
| 2 | 42.94 m | 0.4294 | 0.346 m |
| 3 | 57.16 m | 0.5716 | 0.331 m |
| 4 | 40.49 m | 0.4049 | 0.348 m |
| 5 | 50.10 m | 0.5010 | 0.349 m |

These are two fixed sampling seeds per starting condition, not a large heldout
benchmark. Generation 3 had the highest moving average, about 3.4% above baseline,
but generation 5 finished about 9.4% below baseline. Neither result supports
claiming reliable learning or readiness for a much longer run without diagnosis.

Modal preempted the original job during generation 2 evaluation. Recovery restored
its adapter and Adam optimizer state, preserved completed attempts, repeated only
the unfinished evaluation, and continued to generation 5. The original interrupted
recording retains its original bytes and damaged tail; its verified prefix can be
rendered separately with explicit interrupted metadata. Both Modal apps are stopped.

* [W&B history](https://wandb.ai/skeptrune-org/thunderhill-rl/runs/ozwfci1u)
* [Machine readable results](alpamayo-five-generation-result.json)
* Run: `alpamayo-training-20260924T153854Z`
* Resume source commit: `6cb29c0`
* Recovery rendering source commit: `106056f`
