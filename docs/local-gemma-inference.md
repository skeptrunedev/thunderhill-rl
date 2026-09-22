# Local Gemma evaluation

The current generation 2 adapter completed 1,200 valid actions on the local RTX 2080 Ti: 120 simulator seconds, 921.84 metres, six gates and zero offtrack ticks. It completed Turn 1 and approached Turn 2, with maximum lateral deviation of 0.1611 metres. The driving evaluation took 1,744.61 seconds, about 29 minutes; model loading and the reference comparison are outside that timer. This is a conservative driving test, not a full lap or a training run. The [recorded result](../training/results/rtx2080ti-gemma4-evaluation.json) includes hashes and the execution profile.

All 12 sampled reference prompts produced finite logits and valid controls. Eleven controls matched the saved H100 decisions exactly; one used throttle 14 instead of 15 under local FP16. Peak PyTorch CUDA allocation was 9,490,387,456 bytes. These observations establish practical local inference, not numerical equivalence with the cloud runtime.

The `local-fp16` evaluation profile runs the pinned Gemma 4 E4B text checkpoint and its saved LoRA adapter on the RTX 2080 Ti. It keeps the large per layer embedding table resident in system memory and executes its lookup on the CPU. Only selected vectors move to CUDA. Decoder layers, vocabulary embeddings, and the tied output head remain on the GPU. LoRA parameters stay in FP32. SDPA and the model's normal KV cache remain enabled.

This is unquantized inference, but FP16 compute differs from the H100 BF16 runtime. Pass `--reference-decisions` from the same adapter to measure those differences before driving. The evaluator rejects nonfinite logits and malformed controls, keeps their failure trace, and records the execution profile in its summary. Do not interpret a matching sample as proof of numerical equivalence.

For example, add these options to the existing `training/evaluate_lap.py` invocation:

```bash
--inference-profile local-fp16 \
--reference-decisions PATH_TO_CLOUD_EVALUATION/decisions.jsonl \
--generation 2 \
--max-actions 1200
```

Use the unchanged `--adapter`, `--godot`, and fresh `--output` arguments. The native profile remains the default for cloud training comparisons. Local FP16 does not alter model identity or overwrite adapter weights.

On an 11 GB GPU, run video rendering after inference exits because the resident model leaves little graphics memory for Godot's visual renderer. Keep a concurrent video watcher pointed elsewhere, or publish the completed experiment into its watched directory after the model process exits. Raw simulator recordings are collected during inference regardless of rendering time.
