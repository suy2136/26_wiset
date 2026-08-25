"""Run three independent ABR inference ablations on one NBS-v19 checkpoint.

The matrix contains one NBS-only baseline plus three temporal-selection,
three recent-token-selection, and three MPC-speculative configurations.  It
never trains or mutates the supplied checkpoint and writes partial results
after every completed configuration for safe resume.
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
DEFAULT_EXP_POOL = ABR_ROOT / 'artifacts' / 'exp_pools' / 'exp_pool.pkl'
DEFAULT_OUTPUT = RESULTS_ROOT / 'nbs_v19_three_module_ablation.csv'


EXPERIMENTS = (
    {'name': 'nbs_only', 'family': 'baseline'},
    {'name': 'temporal_k1', 'family': 'temporal', 'max_events': 1},
    {'name': 'temporal_k3', 'family': 'temporal', 'max_events': 3},
    {'name': 'temporal_k4', 'family': 'temporal', 'max_events': 4},
    {'name': 'token_h1', 'family': 'token', 'history_steps': 1},
    {'name': 'token_h5', 'family': 'token', 'history_steps': 5},
    {'name': 'token_h12', 'family': 'token', 'history_steps': 12},
    {
        'name': 'spec_k2_bt0.5_st0.25_rt0.01', 'family': 'speculative',
        'draft_steps': 2, 'buffer_tolerance': 0.5,
        'state_tolerance': 0.25, 'return_tolerance': 0.01,
    },
    {
        'name': 'spec_k2_bt1.0_st0.25_rt0.01', 'family': 'speculative',
        'draft_steps': 2, 'buffer_tolerance': 1.0,
        'state_tolerance': 0.25, 'return_tolerance': 0.01,
    },
    {
        'name': 'spec_k3_bt3.0_st0.40_rt0.01', 'family': 'speculative',
        'draft_steps': 3, 'buffer_tolerance': 3.0,
        'state_tolerance': 0.40, 'return_tolerance': 0.01,
    },
)


def validate_checkpoint(checkpoint_dir, rank_budget):
    required = (
        'adapter_config.json', 'modules_except_plm.bin',
        'nash_rank_allocator.pt', 'checkpoint_metadata.json',
    )
    missing = [name for name in required if not (checkpoint_dir / name).is_file()]
    if not any(
        (checkpoint_dir / name).is_file()
        for name in ('adapter_model.bin', 'adapter_model.safetensors')
    ):
        missing.append('adapter_model.bin or adapter_model.safetensors')
    if missing:
        raise FileNotFoundError(
            f'incomplete NBS v19 checkpoint: {", ".join(missing)}'
        )
    metadata = json.loads(
        (checkpoint_dir / 'checkpoint_metadata.json').read_text(encoding='utf-8')
    )
    if metadata.get('variant') != 'nbs_v19' or metadata.get('seed') != 1:
        raise ValueError('checkpoint must be an NBS v19 seed-1 checkpoint')
    if metadata.get('effective_rank_budget') != rank_budget:
        raise ValueError(
            'checkpoint rank budget does not match --rank-budget: '
            f'{metadata.get("effective_rank_budget")} != {rank_budget}'
        )
    return metadata


def build_command(args, experiment):
    family = experiment['family']
    temporal_selector = 'event-aware' if family == 'temporal' else 'none'
    token_selector = 'recent-timestep' if family == 'token' else 'none'
    draft_steps = experiment.get('draft_steps', 0)
    command = [
        sys.executable, 'run_plm.py', '--test', '--nbs-v19', '--fp16',
        '--seed', '1', '--plm-type', 'llama', '--plm-size', 'base',
        '--plm-dir', str(args.base_model_dir.resolve()),
        '--model-dir', str(args.checkpoint_dir.resolve()),
        '--exp-pool-path', str(args.exp_pool_path.resolve()),
        '--rank', str(args.physical_rank),
        '--nbs-rank-budget', str(args.rank_budget),
        '--nbs-rank-config', str(args.rank_config),
        '--trace', args.trace, '--trace-num', str(args.trace_num),
        '--video', args.video, '--fixed-order',
        '--device', args.device, '--device-out', args.device,
        '--temporal-selector', temporal_selector,
        '--token-selector', token_selector,
        '--speculative-draft-steps', str(draft_steps),
    ]
    if family == 'temporal':
        command.extend([
            '--event-max-events', str(experiment['max_events']),
            '--event-min-spacing', str(args.event_min_spacing),
            '--event-throughput-threshold', str(args.throughput_threshold),
            '--event-buffer-threshold', str(args.buffer_threshold),
            '--event-bitrate-jump-threshold', str(args.bitrate_jump_threshold),
        ])
    elif family == 'token':
        command.extend([
            '--selector-history-steps', str(experiment['history_steps'])
        ])
    elif family == 'speculative':
        command.extend([
            '--speculative-verification-mode', 'greedy',
            '--speculative-buffer-tolerance',
            str(experiment['buffer_tolerance']),
            '--speculative-state-tolerance',
            str(experiment['state_tolerance']),
            '--speculative-return-tolerance',
            str(experiment['return_tolerance']),
        ])
    return command


def newest_metrics(started_at):
    candidates = [
        path for path in RESULTS_ROOT.rglob('selector_metrics.json')
        if path.stat().st_mtime >= started_at
    ]
    if not candidates:
        raise RuntimeError('evaluation produced no selector_metrics.json')
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
    base_reward = baseline.get('mean_reward')
    base_latency = baseline.get('inference_latency_mean_ms')
    for row in rows:
        reward = row.get('mean_reward')
        latency = row.get('inference_latency_mean_ms')
        if isinstance(reward, (int, float)) and isinstance(
            base_reward, (int, float)
        ):
            row['mean_reward_delta_vs_nbs'] = reward - base_reward
            row['mean_reward_change_ratio_vs_nbs'] = (
                0.0 if base_reward == 0 else reward / base_reward - 1.0
            )
        if (
            isinstance(latency, (int, float)) and latency > 0
            and isinstance(base_latency, (int, float)) and base_latency > 0
        ):
            row['inference_speedup_vs_nbs'] = base_latency / latency
            row['inference_latency_reduction_vs_nbs'] = (
                1.0 - latency / base_latency
            )
    return rows


def run_signature(args):
    return {
        'checkpoint_dir': str(args.checkpoint_dir.resolve()),
        'base_model_dir': str(args.base_model_dir.resolve()),
        'exp_pool_path': str(args.exp_pool_path.resolve()),
        'rank_budget': args.rank_budget,
        'physical_rank': args.physical_rank,
        'rank_config': str(args.rank_config),
        'seed': 1,
        'trace': args.trace,
        'trace_num': args.trace_num,
        'video': args.video,
        'event_min_spacing': args.event_min_spacing,
        'throughput_threshold': args.throughput_threshold,
        'buffer_threshold': args.buffer_threshold,
        'bitrate_jump_threshold': args.bitrate_jump_threshold,
        'experiments': [item['name'] for item in EXPERIMENTS],
    }


def load_resume_rows(output, signature):
    rows_path = output.with_suffix('.json')
    manifest_path = output.with_suffix('.manifest.json')
    if not rows_path.is_file() and not manifest_path.is_file():
        return []
    if not rows_path.is_file() or not manifest_path.is_file():
        raise RuntimeError('resume requires both result JSON and manifest JSON')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('signature') != signature:
        raise ValueError('existing result manifest does not match this run')
    return json.loads(rows_path.read_text(encoding='utf-8'))


def write_results(rows, output, signature):
    rows = add_baseline_comparisons(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        'experiment', 'family', 'seed', 'checkpoint_dir', 'rank_budget',
        'mean_reward', 'mean_reward_delta_vs_nbs',
        'mean_reward_change_ratio_vs_nbs', 'qoe_raw_mean',
        'mean_bitrate_mbps', 'total_rebuffer_s',
        'mean_rebuffer_s_per_chunk', 'mean_smoothness_mbps',
        'inference_latency_mean_ms', 'inference_latency_p50_ms',
        'inference_latency_p95_ms', 'inference_speedup_vs_nbs',
        'inference_latency_reduction_vs_nbs', 'original_tokens_mean',
        'selected_tokens_mean', 'token_reduction_ratio',
        'temporal_history_reduction_ratio', 'intra_token_reduction_ratio',
        'acceptance_rate', 'target_plm_calls', 'llm_call_reduction_ratio',
        'draft_attempts', 'drafted_actions', 'accepted_actions',
        'time', 'metrics_path',
    ]
    all_fields = {key for row in rows for key in row}
    fields = [key for key in preferred if key in all_fields]
    fields.extend(sorted(all_fields.difference(fields)))
    with output.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix('.json').write_text(
        json.dumps(rows, indent=2, sort_keys=True), encoding='utf-8'
    )
    output.with_suffix('.manifest.json').write_text(
        json.dumps({
            'signature': signature,
            'completed_experiments': [row['experiment'] for row in rows],
            'requested_experiments': [item['name'] for item in EXPERIMENTS],
        }, indent=2, sort_keys=True),
        encoding='utf-8',
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--base-model-dir', type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument('--exp-pool-path', type=Path, default=DEFAULT_EXP_POOL)
    parser.add_argument('--rank-budget', type=int, default=1536)
    parser.add_argument('--physical-rank', type=int, default=32)
    parser.add_argument(
        '--rank-config', type=Path,
        default=Path('configs/nbs_v19_rank_config.json'),
    )
    parser.add_argument('--event-min-spacing', type=int, default=2)
    parser.add_argument('--throughput-threshold', type=float, default=0.60)
    parser.add_argument('--buffer-threshold', type=float, default=6.0)
    parser.add_argument('--bitrate-jump-threshold', type=int, default=1)
    parser.add_argument('--trace', default='fcc-test')
    parser.add_argument('--trace-num', type=int, default=100)
    parser.add_argument('--video', default='video1')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        '--only', nargs='*', choices=[item['name'] for item in EXPERIMENTS]
    )
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    if args.rank_budget <= 0 or args.physical_rank <= 0:
        parser.error('rank budget and physical rank must be positive')
    selected = [
        item for item in EXPERIMENTS
        if not args.only or item['name'] in args.only
    ]
    signature = run_signature(args)
    metadata = {}
    if not args.dry_run:
        metadata = validate_checkpoint(args.checkpoint_dir, args.rank_budget)
        if not (args.base_model_dir / 'config.json').is_file():
            raise FileNotFoundError(f'base model not found: {args.base_model_dir}')
        if not args.exp_pool_path.is_file():
            raise FileNotFoundError(
                f'experience pool not found: {args.exp_pool_path}'
            )

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
        metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
        rows.append({
            'experiment': experiment['name'],
            'family': experiment['family'],
            'seed': 1,
            'checkpoint_dir': str(args.checkpoint_dir.resolve()),
            'rank_budget': args.rank_budget,
            'nbs_checkpoint_role': metadata.get('role'),
            'metrics_path': str(metrics_path.resolve()),
            **scalar_metrics(metrics),
        })
        write_results(rows, args.output, signature)

    if not args.dry_run:
        print(f'Results saved at: {args.output.resolve()}')


if __name__ == '__main__':
    main()
