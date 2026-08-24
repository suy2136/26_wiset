"""Train ABR NBS v19 once, then run all inference ablations sequentially."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time


ABR_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_MODEL = ABR_ROOT.parent / 'downloaded_plms' / 'llama' / 'base'
DEFAULT_EXP_POOL = ABR_ROOT / 'artifacts' / 'exp_pools' / 'exp_pool.pkl'
DEFAULT_OUTPUT = ABR_ROOT / 'artifacts' / 'results' / 'nbs_v19_inference_matrix.csv'
MODEL_ROOT = ABR_ROOT / 'data' / 'ft_plms'


def build_training_command(args):
    return [
        sys.executable, 'run_plm.py', '--adapt', '--nbs-v19', '--fp16',
        '--seed', '1', '--plm-type', 'llama', '--plm-size', 'base',
        '--plm-dir', str(args.base_model_dir.resolve()),
        '--rank', '32', '--nbs-rank-budget', str(args.rank_budget),
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


def build_inference_command(args, checkpoint_dir):
    command = [
        sys.executable, 'analysis/run_nbs_v19_inference_matrix.py',
        '--checkpoint-dir', str(checkpoint_dir.resolve()),
        '--base-model-dir', str(args.base_model_dir.resolve()),
        '--exp-pool-path', str(args.exp_pool_path.resolve()),
        '--trace', args.test_trace, '--trace-num', str(args.trace_num),
        '--video', args.video, '--device', args.device,
        '--rank-budget', str(args.rank_budget),
        '--output', str(args.output.resolve()),
    ]
    if args.resume_inference:
        command.append('--resume')
    return command


def discover_new_checkpoint(model_root, started_at, rank_budget=512):
    candidates = []
    for metadata_path in model_root.rglob('checkpoint_metadata.json'):
        if metadata_path.stat().st_mtime < started_at:
            continue
        try:
            with metadata_path.open(encoding='utf-8') as stream:
                metadata = json.load(stream)
        except (OSError, ValueError):
            continue
        if metadata.get('variant') != 'nbs_v19' or metadata.get('seed') != 1:
            continue
        if metadata.get('effective_rank_budget') != rank_budget:
            continue
        if metadata.get('role') not in ('best', 'final'):
            continue
        candidates.append((metadata_path, metadata))
    best = [item for item in candidates if item[1].get('role') == 'best']
    selected = best or candidates
    if not selected:
        raise RuntimeError('training finished but no new NBS v19 checkpoint was found')
    metadata_path, _ = max(
        selected, key=lambda item: item[0].stat().st_mtime
    )
    return metadata_path.parent


def state_path(output_path):
    return output_path.with_suffix('.pipeline.json')


def save_pipeline_state(args, checkpoint_dir):
    path = state_path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        'checkpoint_dir': str(checkpoint_dir.resolve()),
        'base_model_dir': str(args.base_model_dir.resolve()),
        'exp_pool_path': str(args.exp_pool_path.resolve()),
        'seed': 1,
        'rank_budget': args.rank_budget,
        'status': 'training_complete',
    }
    with path.open('w', encoding='utf-8') as stream:
        json.dump(state, stream, indent=2, sort_keys=True)


def load_pipeline_checkpoint(args):
    path = state_path(args.output)
    if not path.is_file():
        raise FileNotFoundError(
            f'pipeline state not found: {path}; pass --checkpoint-dir explicitly'
        )
    with path.open(encoding='utf-8') as stream:
        state = json.load(stream)
    if state.get('base_model_dir') != str(args.base_model_dir.resolve()):
        raise ValueError('pipeline state base model does not match this run')
    if state.get('exp_pool_path') != str(args.exp_pool_path.resolve()):
        raise ValueError('pipeline state experience pool does not match this run')
    if state.get('rank_budget') != args.rank_budget:
        raise ValueError('pipeline state rank budget does not match this run')
    return Path(state['checkpoint_dir'])


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model-dir', type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument('--exp-pool-path', type=Path, default=DEFAULT_EXP_POOL)
    parser.add_argument('--checkpoint-dir', type=Path,
                        help='skip training and evaluate this NBS v19 checkpoint')
    parser.add_argument('--train-trace', default='fcc-valid')
    parser.add_argument('--test-trace', default='fcc-test')
    parser.add_argument('--trace-num', type=int, default=100)
    parser.add_argument('--video', default='video1')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--grad-accum-steps', type=int, default=32)
    parser.add_argument('--rank-budget', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--warmup-steps', type=int, default=2000)
    parser.add_argument('--num-epochs', type=int, default=80)
    parser.add_argument('--eval-per-epoch', type=int, default=2)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        '--resume-inference', action='store_true',
        help='skip training and resume the matrix using saved pipeline state',
    )
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.dry_run:
        if not (args.base_model_dir / 'config.json').is_file():
            raise FileNotFoundError(f'base model not found: {args.base_model_dir}')
        if not args.exp_pool_path.is_file():
            raise FileNotFoundError(f'experience pool not found: {args.exp_pool_path}')

    checkpoint_dir = args.checkpoint_dir
    if args.resume_inference and checkpoint_dir is None:
        checkpoint_dir = load_pipeline_checkpoint(args)
    elif checkpoint_dir is None:
        training_command = build_training_command(args)
        print('[training]', shlex.join(training_command), flush=True)
        if args.dry_run:
            checkpoint_dir = MODEL_ROOT / 'NEW_NBS_V19_BEST_MODEL'
        else:
            started_at = time.time() - 1.0
            subprocess.run(training_command, cwd=ABR_ROOT, check=True)
            checkpoint_dir = discover_new_checkpoint(
                MODEL_ROOT, started_at, rank_budget=args.rank_budget
            )
            save_pipeline_state(args, checkpoint_dir)
            print(f'[training] checkpoint selected: {checkpoint_dir}', flush=True)

    inference_command = build_inference_command(args, checkpoint_dir)
    print('[inference-matrix]', shlex.join(inference_command), flush=True)
    if not args.dry_run:
        subprocess.run(inference_command, cwd=ABR_ROOT, check=True)


if __name__ == '__main__':
    main()
