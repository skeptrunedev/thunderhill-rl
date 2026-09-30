# Rev checkpoint assessment

Assessment date: September 30, 2026 (UTC).

The 0.8B does not show sustained improvement from the latest continuations. The 4B completes laps on two training circuits, but remains unreliable across circuits. The evidence does not justify another identical paid training pass yet.

This report covers the Rev telemetry policy and its preserved simulator runs. It is separate from the NVIDIA AlpaGym workflow described in [the training guide](../training/README.md). Recommendations below are conclusions from measured gameplay, not proof of a diagnosed training defect or authorization to start another job.

## 0.8B simulator results

Each evaluation contains 96 attempts: four fixed rolling starts on each of 20 training circuits and four unseen circuits. The three evaluations match all 96 combinations of circuit, starting station and starting speed. They use action temperature 1.0, probability weighted control decoding, pedal gain 1.25, a 240 second episode horizon and seed 0.

| Measurement | s1 at temperature 1.0 | s2 | s3, latest |
| :--- | ---: | ---: | ---: |
| Completed laps | 0/96 | 0/96 | 0/96 |
| Mean legal progress before failure | 874.74 m | 609.37 m | 642.18 m |
| Median legal progress before failure | 649.57 m | 482.91 m | 510.32 m |
| Mean centered progress | 469.54 m | 352.46 m | 360.98 m |
| Mean episode return | 388.29 | 257.03 | 271.13 |
| Mean simulated duration | 24.57 s | 17.04 s | 17.88 s |
| Mean legal progress on training circuits, 80 attempts | 866.54 m | 545.98 m | 606.93 m |
| Mean legal progress on unseen circuits, 16 attempts | 915.72 m | 926.35 m | 818.43 m |
| Offroad terminations | 91 | 88 | 90 |
| Falls | 5 | 8 | 6 |

s3 improves mean legal progress 5.38% over s2, but remains 26.59% below s1 at the same temperature. Unseen circuit progress falls 11.65% from s2 to s3. In matched s2 versus s3 attempts, progress improves on 51 starts and worsens on 45; the median paired gain is only 0.86 m. This small recovery does not establish a reliable improvement trend.

All 96 recording paths exist for each evaluation. Serving hardware and numerical execution differ, and repeated evaluation seeds have not been tested. Worker counts are 4, 16 and 8 respectively. The simulator waits for each control response, so inference latency affects elapsed wall time rather than the duration of a simulated control step.

The original s1 evaluation used temperature 1.3660402567543952 and averaged 992.04 m, with zero completed laps. The s1 evaluation at temperature 1.0 is the more appropriate baseline for s2 and s3.

## What the latest training changed

The recent Rev updates are supervised imitation of specialist teacher controls. They do not directly optimize simulator rewards. Recorded Rev states contribute training examples, but collecting gameplay and then imitating a teacher is distinct from a reward optimizing policy update.

| Checkpoint | Parent | Actual optimizer updates in this run | Training records | Backend |
| :--- | :--- | ---: | ---: | :--- |
| 0.8B s1 | `rev-0.8b-r8` | 20,000 | 160,000 | Modal H100 |
| 0.8B s2 | `rev-0.8b-s1` | 7,500 | 60,000 | Local GPU |
| 0.8B s3 | `rev-0.8b-s2` | 7,500 | 60,000 | Modal H100 |
| 4B s1 | Base model | 20,000 | 160,000 | Modal H100 |

s3 continued from the exact completed local s2 checkpoint. Its `run_config.json` parent result SHA256 matches s2's actual `result.json`. The training, calibration and development dataset hashes are identical for s2 and s3. s3 adds another pass over the same 60,000 training records, with no fresh collected states.

On the same 1,394 development examples:

| Measurement | s2 | s3 |
| :--- | ---: | ---: |
| Continuous steering mean absolute error | 0.326342 | 0.320161 |
| Continuous pedal mean absolute error | 0.404133 | 0.389490 |
| Steering option accuracy | 44.48% | 44.26% |
| Pedal option accuracy | 69.87% | 68.01% |
| Raw development negative log likelihood | 0.858881 | 0.898034 |

Continuous action error improves about 1.9% for steering and 3.6% for pedal. Both control ranges span negative one to positive one. Option accuracy and raw negative log likelihood worsen. These are mixed imitation signals, and simulator outcomes remain the spending criterion.

s2 stores continuous action errors in `result.json`. s3's errors were recovered from its saved `development/rows.json` logits and the identical teacher targets in `data-s2/development.jsonl`, using the probability weighted control calculation in `training/rev_train.py::action_metrics`. They were independently recomputed during this audit. Development sets differ between s1 and s2, so their imitation metrics are not a controlled comparison.

s2 trained in FP32 locally, while s3 trained in BF16 on Modal. Both use learning rate 0.00005 and batch size 8. Total training job wall times were approximately 6 hours 16 minutes for s2 and 68 minutes for s3. Faster execution confirms the handoff worked; it does not establish better riding.

## 4B simulator evidence

The 4B evaluation started 96 attempts, but the configured 3,300 second cutoff interrupted the job before the driver wrote its final evaluation report. Provenance records a failed status and a timeout. The preserved recordings support recovery of 94 terminal outcomes: 88 offroad terminations and six completed laps. Two Portimao attempts have no terminal outcome and are excluded from completed ride comparisons.

All 96 recorded starting conditions match the original 0.8B s1 evaluation, including initial vehicle and track states, physics parameters and timestep, engine, circuit and terrain hashes, and simulator build hashes. The two models use the same recorded action temperature, 1.3660402567543952. Their serving numerical execution differs.

| Matched subset | 4B s1 | Original 0.8B s1 |
| :--- | ---: | ---: |
| Completed laps on training circuits, 80 attempts | 6 | 0 |
| Mean legal progress on training circuits | 1,080.85 m | 993.02 m |
| Completed laps on unseen circuits, 14 finished attempts | 0 | 0 |
| Mean legal progress on those unseen attempts | 758.79 m | 1,051.68 m |

The six laps comprise four at Goiania and two at Brno. Mean training circuit progress improves 8.8%, but the 4B travels farther in only 37 of the 80 training attempts. Mean progress on the matched unseen attempts is 27.9% lower. The gains are concentrated on two circuits rather than broad reliable driving.

There is only one evaluated 4B training checkpoint. This comparison shows a difference between models, not that another 4B training epoch would improve it.

Verified lap episode identities:

| Circuit | Episode identity |
| :--- | :--- |
| Brno | `1790751262_601_2` |
| Brno | `1790751283_688_2` |
| Goiania | `1790751638_891_2` |
| Goiania | `1790751581_862_2` |
| Goiania | `1790751570_833_2` |
| Goiania | `1790751644_920_2` |

To recover the partial evaluation, select recording episode IDs present in `decisions.jsonl`, excluding initialization recordings. Replay every recorded physics transition through the incident rules in `training/alpagym_metrics.py::EpisodeMetrics`. Classify by the first incident, otherwise the terminal transition reason. The simulator's final `terminated` flag alone is insufficient because the Python environment ends an attempt at its first incident. Each of the six successful recordings also contains an explicit `lap_completed` terminal transition.

## Decision

Do not repeat the current 0.8B training recipe merely because Modal runs it faster. The latest checkpoint has no lap completions and remains below the earlier baseline. Investigate the recorded control errors near departures before selecting another training experiment.

The 4B is the stronger candidate for a bounded followup experiment, but the current evidence does not justify an open ended continuation. Fresh recovery experience or reward based learning are possible directions to investigate, not established fixes. Judge any experiment using the same starting conditions, lap completion and progress across unseen circuits. The two interrupted 4B attempts remain outstanding evaluation work.

This assessment and documentation do not launch any additional training or evaluation jobs.

## Evidence locations and identities

Run artifacts remain in local ignored `runs/rev` storage; this committed report does not copy model weights or recordings into Git. Paths below are relative to the repository root.

| Evidence | Artifact path |
| :--- | :--- |
| 0.8B s1 at temperature 1.0 | `runs/rev/rev-0.8b-s1-eval-t1-thelio/{config.json,summary.json,eval/eval.jsonl}` |
| Original 0.8B s1 | `runs/rev/rev-0.8b-s1-eval-s1/{config.json,summary.json,eval/eval.jsonl}` |
| 0.8B s2 | `runs/rev/rev-0.8b-s2-eval-s/{config.json,summary.json,eval/eval.jsonl}` |
| 0.8B s3 | `runs/rev/rev-0.8b-s3-eval-s/{config.json,summary.json,eval/eval.jsonl}` |
| Training results and actual updates | `runs/rev/{rev-0.8b-s1,rev-0.8b-s2,rev-0.8b-s3,rev-4b-s1}/{result.json,checkpoint/training_metrics.json}` |
| Exact s3 parent and dataset identities | `runs/rev/rev-0.8b-s3/run_config.json` |
| Shared s2 and s3 dataset | `runs/rev/data-s2/{summary.json,train.jsonl,calibration.jsonl,development.jsonl}` |
| Development predictions | `runs/rev/{rev-0.8b-s2,rev-0.8b-s3}/development/rows.json` |
| Partial 4B evaluation and recordings | `runs/rev/rev-4b-s1-eval-modal-s1/{provenance.json,config.json,decisions.jsonl,server-final.json,eval/}` |

SHA256 identities recorded during the audit:

```text
0.8B s1 temperature 1.0 eval/eval.jsonl
6248992f9be189479759dca4b9fc1f6834bc7d096adf7620fb9076f60f049e77
0.8B s2 eval/eval.jsonl
5700c505b955b44a730e860083a97552c036f4d94b9b3d64668f268830a25207
0.8B s3 eval/eval.jsonl
270b16da866f70fe3362eb97e7250316f7d88c3af29b25b3295fb56381031052
0.8B s2 result.json (s3 parent result)
4c98330293965374fddd6939a987f411ec5b4297478fd85805e9f4bb1f40902b
4B partial evaluation provenance.json
60483949c66572e10fd7f8b86193cca2aaaa15c35c5dd3eeb1fd628c9c3aceef
```
