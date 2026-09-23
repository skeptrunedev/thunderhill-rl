# Starting checkpoint selection

Catalogue checked on 2026-09-23 using the [OpenRouter models API](https://openrouter.ai/api/v1/models). This is a research shortlist, not a measured driving ranking. Authenticated gameplay screening is implemented in `training/openrouter_benchmark.py`; the shortlist itself is not a measured ranking.

The objective is an instruction tuned checkpoint that already makes useful decisions with road telemetry and native tools. RL should improve its own driving through simulator rewards. Existing FunctionGemma experiments already adapt pretrained weights through LoRA; they do not train a language model from scratch. Supervised imitation remains prohibited.

| OpenRouter ID | Proposed role | Official weights |
| --- | --- | --- |
| `qwen/qwen3.5-9b` | First candidate for balancing a stronger reasoning prior with training and rollout memory | [Qwen3.5 9B](https://huggingface.co/Qwen/Qwen3.5-9B) |
| `mistralai/ministral-3b-2512` | Smaller candidate for eventual local adapter training | [Ministral 3 3B Instruct](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512) |
| `qwen/qwen3.8-27b` | Larger candidate to test whether more starting capability materially helps | [Qwen3.8 27B](https://huggingface.co/Qwen/Qwen3.8-27B) |
| `google/gemma-4-26b-a4b-it` | Gemma family comparison | [Gemma 4 26B A4B](https://huggingface.co/google/gemma-4-26B-A4B-it) |

All four IDs currently advertise tools and tool choice support, and their official weights are published under Apache 2.0. None is proven to control this motorcycle. Local memory and native training compatibility must be measured before selecting a training configuration. Ministral's published instruction checkpoint uses FP8, which the RTX 2080 Ti does not natively support. Gemma 26B A4B has roughly 25.2B total parameters despite only 3.8B being active per token; active count is not its weight memory requirement.

[Gemma 4 E4B](https://huggingface.co/google/gemma-4-E4B-it) exists as downloadable weights but was absent from the current OpenRouter catalogue. Qwen3.5 4B was also absent. Do not invent hosted IDs for them. They can be considered through a separately hosted endpoint.

## Evaluation before training

1. Test the unmodified instruction checkpoints through native API tool calls using the same simulator, road telemetry, action cadence, stall rules and audited reward. Preserve every attempt and its video job. Model formatting mistakes must not be repaired by a fallback controller.
2. Run a small screening set, then compare finalists over repeated attempts and longer simulation budgets. Track median and mean legal progress, completed laps, stalls, track violations, reward, tool validity and decision latency. Do not select by best attempt alone.
3. Separate deterministic and sampled evaluations. Record temperature, reasoning settings, requested and resolved model, and provider. Do not inherit the tiny model's temperature 3 automatically. Provider seed support does not guarantee deterministic outputs.
4. Reproduce the winner's behavior with its actual downloaded checkpoint and native chat template before training. API provider precision and serialization can differ from local execution.
5. Apply reward driven adapter RL to that checkpoint using its own newly collected gameplay. OpenRouter responses are evaluation evidence, not supervised targets or teacher demonstrations.

If stronger models also fail to make useful progress before training, investigate observation clarity and control semantics before assuming model scale or longer RL will solve the task.

The existing `LapEpisode` implementation can supply reward auditing, stalled attempt termination, recording and video provenance. A remote adapter should consume real API `tool_calls` and retain the most recent assistant call plus actual tool response. It must identify a remote request configuration hash separately from a verified local weight hash. It must not fabricate Gemma token delimiters or behavior log probabilities for API outputs. See [OpenRouter native tool documentation](https://openrouter.ai/docs/guides/features/tool-calling).

Authenticated driving evaluation accepts `OPENROUTER_API_KEY` or a private file through `--key-file`. Credentials must never appear in prompts, recordings, committed files or logs.

The default `--routing auto` sends the model directly to OpenRouter without a provider object or endpoint selection. OpenRouter may change providers between decisions, and each response preserves its actual provider identity. Sampling seeds are omitted in this mode, and native tool choice is automatic; missing or malformed calls still fail validation. Requested reasoning settings are recorded but provider support may vary. This mode uses [OpenRouter's default routing](https://openrouter.ai/docs/guides/routing/provider-selection).

For a controlled provider comparison, `--routing pinned` queries endpoint capabilities, requires the full parameter intersection, pins one endpoint and disables provider fallback. API failures are archived separately from driving failures in either mode. No controls are repaired.

Run two screening attempts per model with `--episodes 2 --seconds 30 --temperature 0.6 --reasoning disabled`, supplying the selected model IDs, output directory, Godot executable and private credential path. Every completed attempt updates `campaign.json` and each model's aggregate ledger. Raw API responses, decision latency, effective settings, simulator recordings and video jobs are retained. These hosted evaluations are explicitly ineligible for training, and their configuration fingerprints do not claim to identify hosted weights.
