"""Evaluate one ABR NBS-v19 checkpoint with fixed inference ablations.

The runner never trains or mutates the checkpoint.  Every row uses seed 1 and
the same model weights; only selector/speculative inference parameters differ.
"""

import argparse
import csv
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time


ABR_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ABR_ROOT / 'artifacts' / 'results'
DEFAULT_BASE_MODEL = ABR_ROOT.parent / 'downloaded_plms' / 'llama' / 'base'
DEFAULT_OUTPUT = ABR_ROOT / 'artifacts' / 'results' / 'nbs_v19_inference_matrix.csv'


EXPERIMENTS = (
    {'name': 'nbs_only', 'selector': 'none', 'history': None, 'draft_steps': 0},
    {'name': 'selector_h5', 'selector': 'recent-timestep', 'history': 5, 'draft_steps': 0},
    {'name': 'selector_h10', 'selector': 'recent-timestep', 'history': 10, 'draft_steps': 0},
    {'name': 'selector_h15', 'selector': 'recent-timestep', 'history': 15, 'draft_steps': 0},
    {'name': 'speculative_k2', 'selector': 'none', 'history': None, 'draft_steps': 2},
    {'name': 'speculative_k3', 'selector': 'none', 'history': None, 'draft_steps': 3},
    {'name': 'speculative_k4', 'selector': 'none', 'history': None, 'draft_steps': 4},
    {'name': 'combined_h5_k2', 'selector': 'recent-timestep', 'history': 5, 'draft_steps': 2},
    {'name': 'combined_h10_k3', 'selector': 'recent-timestep', 'history': 10, 'draft_steps': 3},
)


def validate_checkpoint(checkpoint_dir, rank_budget=512):
    required = (
        'adapter_config.json',
        'modules_except_plm.bin',
        'nash_rank_allocator.pt',
    )
    missing = [name for name in required if not (checkpoint_dir / name).is_file()]
    if not any(
        (checkpoint_dir / name).is_file()
        for name in ('adapter_model.bin', 'adapter_model.safetensors')
    ):
        missing.append('adapter_model.bin or adapter_model.safetensors')
    if missing:
        raise FileNotFoundError(
            f'incomplete NBS v19 checkpoint {checkpoint_dir}: {", ".join(missing)}'
        )

    metadata_path = checkpoint_dir / 'checkpoint_metadata.json'
    metadata = {}
    if metadata_path.is_file():
        with metadata_path.open(encoding='utf-8') as stream:
            metadata = json.load(stream)
        if metadata.get('variant') != 'nbs_v19':
            raise ValueError('checkpoint metadata is not tagged nbs_v19')
        if metadata.get('seed') != 1:
            raise ValueError('the fixed experiment matrix requires a seed-1 checkpoint')
        if metadata.get('effective_rank_budget') != rank_budget:
            raise ValueError(
                'checkpoint rank budget does not match --rank-budget: '
                f'{metadata.get("effective_rank_budget")} != {rank_budget}'
            )
    return metadata


def build_command(args, experiment):
    command = [
        sys.executable, 'run_plm.py', '--test', '--nbs-v19', '--fp16',
        '--plm-type', 'llama', '--plm-size', 'base', '--rank', '32',
        '--nbs-rank-budget', str(args.rank_budget), '--seed', '1',
        '--plm-dir', str(args.base_model_dir.resolve()),
        '--model-dir', str(args.checkpoint_dir.resolve()),
        '--exp-pool-path', str(args.exp_pool_path.resolve()),
        '--device', args.device, '--device-out', args.device,
        '--trace', args.trace, '--trace-num', str(args.trace_num),
        '--video', args.video, '--fixed-order',
        '--token-selector', experiment['selector'],
        '--speculative-draft-steps', str(experiment['draft_steps']),
        '--speculative-verification-mode', 'greedy',
        '--speculative-buffer-tolerance', str(args.buffer_tolerance),
        '--speculative-state-tolerance', str(args.state_tolerance),
        '--speculative-return-tolerance', str(args.return_tolerance),
    ]
    if experiment['history'] is not None:
        command.extend([
            '--selector-history-steps', str(experiment['history'])
        ])
    return command


def newest_metrics(started_at):
    candidates = [
        path for path in RESULTS_ROOT.rglob('selector_metrics.json')
        if path.stat().st_mtime >= started_at
    ]
    if not candidates:
        raise RuntimeError('evaluation finished without selector_metrics.json')
    return max(candidates, key=lambda path: path.stat().st_mtime)


def scalar_metrics(metrics):
    return {
        key: value for key, value in metrics.items()
        if value is None or isinstance(value, (str, int, float, bool))
    }


def add_baseline_comparisons(rows):
    baseline = next(
        (row for row in rows if row['experiment'] == 'nbs_only'), None
    )
    if baseline is None:
        return rows
    baseline_reward = baseline.get('mean_reward')
    baseline_latency = baseline.get('inference_latency_mean_ms')
    baseline_time = baseline.get('time')
    for row in rows:
        reward = row.get('mean_reward')
        latency = row.get('inference_latency_mean_ms')
        elapsed = row.get('time')
        if isinstance(reward, (int, float)) and isinstance(
            baseline_reward, (int, float)
        ):
            row['mean_reward_delta_vs_nbs'] = reward - baseline_reward
        if (
            isinstance(latency, (int, float)) and latency > 0
            and isinstance(baseline_latency, (int, float))
            and baseline_latency > 0
        ):
            row['inference_speedup_vs_nbs'] = baseline_latency / latency
            row['inference_latency_reduction_vs_nbs'] = (
                1.0 - latency / baseline_latency
            )
        if (
            isinstance(elapsed, (int, float)) and elapsed > 0
            and isinstance(baseline_time, (int, float)) and baseline_time > 0
        ):
            row['test_time_speedup_vs_nbs'] = baseline_time / elapsed
    return rows


def run_signature(args):
    return {
        'checkpoint_dir': str(args.checkpoint_dir.resolve()),
        'base_model_dir': str(args.base_model_dir.resolve()),
        'exp_pool_path': str(args.exp_pool_path.resolve()),
        'seed': 1,
        'rank_budget': args.rank_budget,
        'trace': args.trace,
        'trace_num': args.trace_num,
        'video': args.video,
        'buffer_tolerance': args.buffer_tolerance,
        'state_tolerance': args.state_tolerance,
        'return_tolerance': args.return_tolerance,
    }


def load_resume_rows(output_path, signature):
    rows_path = output_path.with_suffix('.json')
    manifest_path = output_path.with_suffix('.manifest.json')
    if not rows_path.is_file() and not manifest_path.is_file():
        return []
    if not rows_path.is_file() or not manifest_path.is_file():
        raise RuntimeError('resume requires both result JSON and manifest JSON')
    with manifest_path.open(encoding='utf-8') as stream:
        manifest = json.load(stream)
    if manifest.get('signature') != signature:
        raise ValueError('existing result manifest does not match this run')
    with rows_path.open(encoding='utf-8') as stream:
        rows = json.load(stream)
    return rows


def write_results(rows, output_path, signature=None):
    rows = add_baseline_comparisons(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        'experiment', 'seed', 'checkpoint_dir', 'metrics_path',
        'mean_reward', 'mean_reward_delta_vs_nbs', 'time',
        'test_time_speedup_vs_nbs', 'inference_latency_mean_ms',
        'inference_speedup_vs_nbs', 'inference_latency_reduction_vs_nbs',
        'inference_latency_p95_ms', 'original_tokens_mean',
        'selected_tokens_mean', 'token_reduction_ratio', 'draft_attempts',
        'drafted_actions', 'accepted_actions', 'acceptance_rate',
        'fallback_calls', 'target_plm_calls', 'llm_call_reduction_ratio',
    ]
    all_fields = {key for row in rows for key in row}
    fields = [key for key in preferred if key in all_fields]
    fields.extend(sorted(all_fields.difference(fields)))
    with output_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with output_path.with_suffix('.json').open('w', encoding='utf-8') as stream:
        json.dump(rows, stream, indent=2, sort_keys=True)
    if signature is not None:
        manifest = {
            'signature': signature,
            'completed_experiments': [row['experiment'] for row in rows],
            'requested_experiments': [item['name'] for item in EXPERIMENTS],
        }
        with output_path.with_suffix('.manifest.json').open(
            'w', encoding='utf-8'
        ) as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--base-model-dir', type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument(
        '--exp-pool-path', type=Path,
        default=ABR_ROOT / 'artifacts' / 'exp_pools' / 'exp_pool.pkl',
    )
    parser.add_argument('--trace', default='fcc-test')
    parser.add_argument('--trace-num', type=int, default=100)
    parser.add_argument('--video', default='video1')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--rank-budget', type=int, default=512)
    parser.add_argument('--buffer-tolerance', type=float, default=1.0)
    parser.add_argument('--state-tolerance', type=float, default=0.25)
    parser.add_argument('--return-tolerance', type=float, default=0.01)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        '--only', choices=[item['name'] for item in EXPERIMENTS], nargs='*',
        help='run only the named configurations (default: all nine)',
    )
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument(
        '--resume', action='store_true',
        help='reuse matching partial JSON/manifest output and skip completed runs',
    )
    args = parser.parse_args()

    selected = [
        item for item in EXPERIMENTS
        if not args.only or item['name'] in args.only
    ]
    signature = run_signature(args)
    checkpoint_metadata = {}
    if not args.dry_run:
        checkpoint_metadata = validate_checkpoint(
            args.checkpoint_dir, rank_budget=args.rank_budget
        )
        if not (args.base_model_dir / 'config.json').is_file():
            raise FileNotFoundError(f'base model not found: {args.base_model_dir}')
        if not args.exp_pool_path.is_file():
            raise FileNotFoundError(f'experience pool not found: {args.exp_pool_path}')

    rows = (
        load_resume_rows(args.output, signature)
        if args.resume and not args.dry_run else []
    )
    completed = {row['experiment'] for row in rows}
    for experiment in selected:
        if experiment['name'] in completed:
            print(f"[{experiment['name']}] already complete; skipping", flush=True)
            continue
        command = build_command(args, experiment)
        print(f"[{experiment['name']}] {shlex.join(command)}", flush=True)
        if args.dry_run:
            continue
        started_at = time.time() - 1.0
        subprocess.run(command, cwd=ABR_ROOT, check=True)
        metrics_path = newest_metrics(started_at)
        with metrics_path.open(encoding='utf-8') as stream:
            metrics = json.load(stream)
        row = {
            'experiment': experiment['name'],
            'seed': 1,
            'checkpoint_dir': str(args.checkpoint_dir.resolve()),
            'metrics_path': str(metrics_path.resolve()),
            'nbs_checkpoint_role': checkpoint_metadata.get('role'),
            'nbs_effective_rank_budget': checkpoint_metadata.get(
                'effective_rank_budget', args.rank_budget
            ),
            **scalar_metrics(metrics),
        }
        rows.append(row)
        write_results(rows, args.output, signature=signature)

    if not args.dry_run:
        print(f'Results saved at: {args.output.resolve()}')


if __name__ == '__main__':
    main()
