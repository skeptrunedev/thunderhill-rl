"""Bounded CUDA allocation audit using recorded prompts and compiled native calls.

No optimizer steps, supervised targets, simulator actions, or graph resets occur.
"""
import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer, set_seed

from batched_policy import BatchedPolicy, cuda_memory_evidence
from model_runtime import PolicyRoadTelemetry, load_base, read_spec
from trajectory_update import TrajectoryConfig, TrajectoryUpdater
from train_full_lap_grpo import save_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--decisions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=4)
    parser.add_argument('--calls-per-round', type=int, default=20)
    parser.add_argument('--roundtrip-adapter', action='store_true')
    parser.add_argument('--mark-step', action='store_true')
    parser.add_argument('--profile-backward', action='store_true')
    parser.add_argument('--accumulate-collection', type=Path)
    parser.add_argument('--touch-parameter-versions', action='store_true',
                        help='Increment tensor version counters without changing weights')
    args = parser.parse_args()
    if min(args.rounds, args.calls_per_round) < 1:
        parser.error('Positive round and call counts required')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        torch.set_num_threads(4)
        set_seed(73)
        spec = read_spec(args.adapter)
        tokenizer = AutoTokenizer.from_pretrained(args.adapter, padding_side='left')
        model = PeftModel.from_pretrained(load_base(spec), args.adapter, is_trainable=True)
        road = PolicyRoadTelemetry(spec, tokenizer)
        policy = BatchedPolicy(model, tokenizer, compile_inference=True,
                               native_tools=road.native_tools, temperature=3)
        decisions = [json.loads(line) for line in args.decisions.read_text().splitlines()]
        prompts = [tokenizer.decode(row['prompt_ids'], skip_special_tokens=False) for row in decisions]
        updater = TrajectoryUpdater(model, tokenizer, args.output.with_suffix('.updates'),
                                    TrajectoryConfig(max_actions=300, temperature=3,
                                                     max_completion_length=road.native_tools.max_completion_length),
                                    constraints=policy.constraints)
        episode = {'reward': 0.0, 'decisions': [dict(prompt_ids=row['prompt_ids'],
                    completion_ids=row['completion_ids'], old_per_token_logps=row['behavior_logprobs'])
                    for row in decisions]}
        def record(event, **values):
            report = dict(event=event, **values, memory=cuda_memory_evidence(model.device))
            print(json.dumps(report), flush=True)
            stream.write(json.dumps(report) + '\n')
            stream.flush()
        episodes = [episode, episode]
        if args.accumulate_collection:
            collection = json.loads(args.accumulate_collection.read_text())
            episodes = []
            for wave in sorted(args.accumulate_collection.parent.glob('wave-*')):
                for path in sorted(wave.glob('rollout-*/decisions.jsonl')):
                    if path.parent.name == 'rollout-0000':
                        continue
                    rows = [json.loads(line) for line in path.read_text().splitlines()]
                    summary = collection['rollouts'][len(episodes)]
                    episodes.append(dict(reward_components=summary['reward_components'], decisions=[
                        dict(prompt_ids=row['prompt_ids'], completion_ids=row['completion_ids'],
                             old_per_token_logps=row['behavior_logprobs']) for row in rows]))
        record('loaded')
        for round_index in range(args.rounds):
            for call in range(args.calls_per_round):
                offset = round_index * args.calls_per_round + call
                if args.mark_step:
                    torch.compiler.cudagraph_mark_step_begin()
                policy.generate([prompts[(offset + i) % len(prompts)] for i in range(4)], greedy_indices=(0,))
                if call % 5 == 0:
                    record('generation_call', round=round_index, call=offset + 1)
            record('generated', round=round_index, calls=(round_index + 1) * args.calls_per_round)
            if args.profile_backward:
                report = updater.profile_microbatches(episodes, candidates=(1, 2, 4, 8))
                record('profile_backward', round=round_index, training_profile=report)
            if args.accumulate_collection:
                if round_index == 0:
                    rows, _ = updater.rows(episodes)
                    batch = updater.batch(rows[:updater.config.microbatch_size])
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
                        loss = updater.trainer.trajectory_loss(model, batch)
                        loss.backward()
                    record('attention_backend', operators=[entry.key for entry in trace.key_averages()
                                                          if 'attention' in entry.key])
                    del loss, batch
                    updater.optimizer.zero_grad(set_to_none=True)
                audit = updater.accumulate(episodes)
                updater.optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                record('accumulated', round=round_index, audit=audit)
            if args.touch_parameter_versions:
                with torch.no_grad():
                    for parameter in model.parameters():
                        if parameter.requires_grad:
                            parameter.add_(0)
                record('parameter_versions', round=round_index)
            if args.roundtrip_adapter:
                save_checkpoint(model, tokenizer, spec, updater,
                                args.output.with_suffix('.checkpoints') / str(round_index), prompts[0])
                record('adapter_roundtrip', round=round_index)
            model.train()
            model.eval()
        record('complete', compile_qualified=policy.compile_qualified)


if __name__ == '__main__':
    main()
