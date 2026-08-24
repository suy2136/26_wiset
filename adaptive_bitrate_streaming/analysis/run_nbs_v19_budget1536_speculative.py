"""Train ABR NBS v19 at budget 1536 and evaluate 40 MPC drafts.

The pipeline uses seed 1 throughout, writes every completed evaluation
immediately, and compares the trained NBS checkpoint with the official
NetLLM rank-128 LoRA under the same traces.
"""

import argparse
import csv
import itertools
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time


ABR_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ABR_ROOT / 'artifacts' / 'results'
MODEL_ROOT = ABR_ROOT / 'data' / 'ft_plms'
DEFAULT_BASE_MODEL = ABR_ROOT.parent / 'downloaded_plms' / 'llama' / 'base'
DEFAULT_EXP_POOL = ABR_ROOT / 'artifacts' / 'exp_pools' / 'exp_pool.pkl'
DEFAULT_OFFICIAL_LORA = Path(
    '/workspace/abr_checkpoint_download/extracted/try_llama2_7b'
)
DEFAULT_OUTPUT = RESULTS_ROOT / 'nbs_v19_budget1536_speculative_sweep.csv'

RANK_BUDGET = 1536
DRAFT_STEPS = (2, 3)
BUFFER_TOLERANCES = (0.5, 1.0, 1.5, 2.0, 3.0)
STATE_TOLERANCES = (0.10, 0.25, 0.40, 0.50)
RETURN_TOLERANCE = 0.01


def speculative_configurations():
    return [
        {
            'experiment': f'spec_k{k}_bt{bt}_st{st:.2f}_rt{RETURN_TOLERANCE}',
            'draft_steps': k,
            'buffer_tolerance': bt,
            'state_tolerance': st,
            'return_tolerance': RETURN_TOLERANCE,
        }
        for k, bt, st in itertools.product(
            DRAFT_STEPS, BUFFER_TOLERANCES, STATE_TOLERANCES
        )
    ]


def build_training_command(args):
    return [
        sys.executable, 'run_plm.py', '--adapt', '--nbs-v19', '--fp16',
        '--seed', '1', '--plm-type', 'llama', '--plm-size', 'base',
        '--plm-dir', str(args.base_model_dir.resolve()),
        '--rank', '32', '--nbs-rank-budget', str(RANK_BUDGET),
        '--nbs-rank-config', 'configs/nbs_v19_rank_config.json',
        '--token-selector', 'none', '--speculative-draft-steps', '0',
        '--exp-pool-path', str(args.exp_pool_path.resolve()),
        '--trace', args.train_trace, '--trace-num', str(args.trace_num),
        '--video', args.video, '--fixed-order',
        '--device', args.device, '--device-out', args.device,
        '--grad-accum-steps', str(args.grad_accum_steps),
        '--lr', str(args.lr), '--warmup-steps', str(args.warmup_steps),
        '--num-epochs', str(args.num_epochs),
        '--eval-per-epoch', str(args.eval_per_epoch),
    ]


def common_test_command(args):
    return [
        sys.executable, 'run_plm.py', '--test', '--fp16', '--seed', '1',
        '--plm-type', 'llama', '--plm-size', 'base',
        '--plm-dir', str(args.base_model_dir.resolve()),
        '--exp-pool-path', str(args.exp_pool_path.resolve()),
        '--trace', args.test_trace, '--trace-num', str(args.trace_num),
        '--video', args.video, '--fixed-order',
        '--device', args.device, '--device-out', args.device,
        '--token-selector', 'none',
    ]


def build_nbs_test_command(args, checkpoint_dir, config=None):
    config = config or {
        'draft_steps': 0,
        'buffer_tolerance': 1.0,
        'state_tolerance': 0.25,
        'return_tolerance': RETURN_TOLERANCE,
    }
    return [
        *common_test_command(args), '--nbs-v19', '--rank', '32',
        '--nbs-rank-budget', str(RANK_BUDGET),
        '--nbs-rank-config', 'configs/nbs_v19_rank_config.json',
        '--model-dir', str(checkpoint_dir.resolve()),
        '--speculative-draft-steps', str(config['draft_steps']),
        '--speculative-verification-mode', 'greedy',
        '--speculative-buffer-tolerance', str(config['buffer_tolerance']),
        '--speculative-state-tolerance', str(config['state_tolerance']),
        '--speculative-return-tolerance', str(config['return_tolerance']),
    ]


def build_official_test_command(args):
    return [
        *common_test_command(args), '--rank', '128',
        '--model-dir', str(args.official_lora_dir.resolve()),
        '--speculative-draft-steps', '0',
    ]


def discover_checkpoint(started_at):
    candidates = []
    for path in MODEL_ROOT.rglob('checkpoint_metadata.json'):
        if path.stat().st_mtime < started_at:
            continue
        try:
            metadata = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if (
            metadata.get('variant') == 'nbs_v19'
            and metadata.get('seed') == 1
            and metadata.get('effective_rank_budget') == RANK_BUDGET
            and metadata.get('role') in ('best', 'final')
        ):
            candidates.append((path, metadata))
    preferred = [item for item in candidates if item[1].get('role') == 'best']
    selected = preferred or candidates
    if not selected:
        raise RuntimeError('no new budget-1536 NBS best/final checkpoint found')
    return max(selected, key=lambda item: item[0].stat().st_mtime)[0].parent


def validate_checkpoint(checkpoint_dir):
    required = ('adapter_config.json', 'modules_except_plm.bin',
                'nash_rank_allocator.pt', 'checkpoint_metadata.json')
    missing = [name for name in required if not (checkpoint_dir / name).is_file()]
    if not any((checkpoint_dir / name).is_file() for name in (
        'adapter_model.bin', 'adapter_model.safetensors'
    )):
        missing.append('adapter_model.bin or adapter_model.safetensors')
    if missing:
        raise FileNotFoundError(f'incomplete NBS checkpoint: {", ".join(missing)}')
    metadata = json.loads(
        (checkpoint_dir / 'checkpoint_metadata.json').read_text(encoding='utf-8')
    )
    if metadata.get('effective_rank_budget') != RANK_BUDGET:
        raise ValueError('NBS checkpoint does not use rank budget 1536')
    return metadata


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


def add_comparisons(rows):
    nbs = next((row for row in rows if row['experiment'] == 'nbs_only'), None)
    official = next(
        (row for row in rows if row['experiment'] == 'netllm_official_lora'),
        None,
    )
    for row in rows:
        for suffix, baseline in (('nbs', nbs), ('official', official)):
            if baseline is None:
                continue
            reward = row.get('mean_reward')
            base_reward = baseline.get('mean_reward')
            latency = row.get('inference_latency_mean_ms')
            base_latency = baseline.get('inference_latency_mean_ms')
            if isinstance(reward, (int, float)) and isinstance(
                base_reward, (int, float)
            ):
                row[f'mean_reward_delta_vs_{suffix}'] = reward - base_reward
            if (
                isinstance(latency, (int, float)) and latency > 0
                and isinstance(base_latency, (int, float)) and base_latency > 0
            ):
                row[f'inference_speedup_vs_{suffix}'] = base_latency / latency
    return rows


def write_results(rows, output, signature):
    rows = add_comparisons(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        'experiment', 'mean_reward', 'mean_reward_delta_vs_nbs',
        'mean_reward_delta_vs_official', 'qoe_raw_mean', 'mean_bitrate_mbps',
        'total_rebuffer_s', 'mean_rebuffer_s_per_chunk',
        'mean_smoothness_mbps', 'time', 'inference_latency_mean_ms',
        'inference_latency_p50_ms', 'inference_latency_p95_ms',
        'inference_speedup_vs_nbs', 'inference_speedup_vs_official',
        'acceptance_rate', 'llm_call_reduction_ratio', 'target_plm_calls',
        'draft_attempts', 'drafted_actions', 'accepted_actions',
        'state_mismatch_fallbacks', 'buffer_mismatch_fallbacks',
        'return_mismatch_fallbacks', 'feature_mismatch_fallbacks',
        'metrics_path', 'checkpoint_dir', 'rank_budget', 'seed',
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
            'requested_speculative_configurations': 40,
            'nbs_training_artifacts': signature.get('nbs_training_artifacts'),
        }, indent=2, sort_keys=True),
        encoding='utf-8',
    )


def load_resume(output, signature):
    json_path = output.with_suffix('.json')
    manifest_path = output.with_suffix('.manifest.json')
    if not json_path.exists() and not manifest_path.exists():
        return []
    if not json_path.exists() or not manifest_path.exists():
        raise RuntimeError('resume requires both result and manifest JSON files')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('signature') != signature:
        raise ValueError('existing sweep manifest does not match this run')
    return json.loads(json_path.read_text(encoding='utf-8'))


def resume_checkpoint(output):
    manifest_path = output.with_suffix('.manifest.json')
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f'resume manifest not found: {manifest_path}; pass --checkpoint-dir'
        )
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    checkpoint = manifest.get('signature', {}).get('checkpoint_dir')
    if not checkpoint:
        raise ValueError('resume manifest has no NBS checkpoint path')
    return Path(checkpoint)


def run_evaluation(name, command, checkpoint_dir, output, rows, signature):
    print(f'[{name}] {shlex.join(command)}', flush=True)
    started_at = time.time() - 1.0
    subprocess.run(command, cwd=ABR_ROOT, check=True)
    metrics_path = newest_metrics(started_at)
    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    rows.append({
        'experiment': name,
        'seed': 1,
        'rank_budget': RANK_BUDGET if name != 'netllm_official_lora' else 8192,
        'checkpoint_dir': str(checkpoint_dir.resolve()),
        'metrics_path': str(metrics_path.resolve()),
        **scalar_metrics(metrics),
    })
    write_results(rows, output, signature)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model-dir', type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument('--exp-pool-path', type=Path, default=DEFAULT_EXP_POOL)
    parser.add_argument('--official-lora-dir', type=Path, default=DEFAULT_OFFICIAL_LORA)
    parser.add_argument('--checkpoint-dir', type=Path,
                        help='skip training and evaluate this budget-1536 checkpoint')
    parser.add_argument('--train-trace', default='fcc-valid')
    parser.add_argument('--test-trace', default='fcc-test')
    parser.add_argument('--trace-num', type=int, default=100)
    parser.add_argument('--video', default='video1')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--grad-accum-steps', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--warmup-steps', type=int, default=2000)
    parser.add_argument('--num-epochs', type=int, default=80)
    parser.add_argument('--eval-per-epoch', type=int, default=2)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--skip-official', action='store_true')
    parser.add_argument('--skip-plots', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if len(speculative_configurations()) != 40:
        raise RuntimeError('the fixed speculative grid must contain 40 runs')
    if not args.dry_run:
        if not (args.base_model_dir / 'config.json').is_file():
            raise FileNotFoundError(f'base model not found: {args.base_model_dir}')
        if not args.exp_pool_path.is_file():
            raise FileNotFoundError(f'experience pool not found: {args.exp_pool_path}')
        if not args.skip_official and not args.official_lora_dir.is_dir():
            raise FileNotFoundError(f'official LoRA not found: {args.official_lora_dir}')

    checkpoint_dir = args.checkpoint_dir
    if args.resume and checkpoint_dir is None:
        checkpoint_dir = resume_checkpoint(args.output)
    if checkpoint_dir is None:
        command = build_training_command(args)
        print(f'[training] {shlex.join(command)}', flush=True)
        if args.dry_run:
            checkpoint_dir = MODEL_ROOT / 'NEW_BUDGET1536_BEST_MODEL'
        else:
            started_at = time.time() - 1.0
            subprocess.run(command, cwd=ABR_ROOT, check=True)
            checkpoint_dir = discover_checkpoint(started_at)
            print(f'[training] selected {checkpoint_dir}', flush=True)
    if not args.dry_run:
        validate_checkpoint(checkpoint_dir)

    signature = {
        'checkpoint_dir': str(checkpoint_dir.resolve()),
        'official_lora_dir': None if args.skip_official else str(
            args.official_lora_dir.resolve()
        ),
        'base_model_dir': str(args.base_model_dir.resolve()),
        'exp_pool_path': str(args.exp_pool_path.resolve()),
        'rank_budget': RANK_BUDGET,
        'seed': 1,
        'test_trace': args.test_trace,
        'trace_num': args.trace_num,
        'video': args.video,
        'nbs_training_artifacts': {
            'rank_trajectory_and_nash_statistics': str((
                checkpoint_dir.parent / 'early_stop_-1_checkpoint'
                / 'nbs_rank_diagnostics.csv'
            ).resolve()),
            'numeric_events': str((
                checkpoint_dir.parent / 'early_stop_-1_checkpoint'
                / 'nbs_numeric_events.jsonl'
            ).resolve()),
            'training_console': str((
                checkpoint_dir.parent / 'early_stop_-1_console.log'
            ).resolve()),
            'allocator_checkpoint': str((
                checkpoint_dir / 'nash_rank_allocator.pt'
            ).resolve()),
        },
    }
    rows = load_resume(args.output, signature) if args.resume and not args.dry_run else []
    completed = {row['experiment'] for row in rows}
    jobs = [('nbs_only', build_nbs_test_command(args, checkpoint_dir), checkpoint_dir)]
    if not args.skip_official:
        jobs.append((
            'netllm_official_lora', build_official_test_command(args),
            args.official_lora_dir,
        ))
    jobs.extend((
        config['experiment'], build_nbs_test_command(args, checkpoint_dir, config),
        checkpoint_dir,
    ) for config in speculative_configurations())

    for name, command, job_checkpoint in jobs:
        if name in completed:
            print(f'[{name}] already complete; skipping', flush=True)
            continue
        if args.dry_run:
            print(f'[{name}] {shlex.join(command)}', flush=True)
            continue
        run_evaluation(
            name, command, job_checkpoint, args.output, rows, signature
        )

    if not args.dry_run and not args.skip_plots:
        plot_dir = args.output.parent / f'{args.output.stem}_plots'
        plot_command = [
            sys.executable, 'analysis/plot_nbs_v19_inference_results.py',
            str(args.output.resolve()), '--output-dir', str(plot_dir.resolve()),
            '--prefix', 'nbs_v19_budget1536_speculative',
        ]
        print(f'[plots] {shlex.join(plot_command)}', flush=True)
        subprocess.run(plot_command, cwd=ABR_ROOT, check=True)


if __name__ == '__main__':
    main()
