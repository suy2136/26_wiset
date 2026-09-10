"""Validation-only search for the training-free VP patch selector."""
import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NBS = Path(
    'viewport_prediction/data/ft_plms/llama_base_low_rank_adalora_nbs_v19/'
    'freeze_plm_False/multimodal_none/Jin2022/5Hz/20260821_204908/'
    'his_10_fut_20_ss_15_epochs_4_bs_32_lr_0.0002_seed_1_rank_32_'
    'scheduled_sampling_False/best_ar_model'
)


PARAMETER_SPACE = {
    'top_k': [4, 6, 8, 10, 12],
    'velocity_window': [2, 3, 5],
    'horizon_scale': [0.5, 1.0, 1.5, 2.0],
    'acceleration_weight': [0.0, 0.5, 1.0],
    'uncertainty_deg': [0.0, 20.0, 40.0],
    'uncertainty_growth': [0.0, 1.0, 2.0],
}
PARAMETER_KEYS = tuple(PARAMETER_SPACE)


def base_candidate():
    return {
        'top_k': 8,
        'velocity_window': 3,
        'horizon_scale': 1.0,
        'acceleration_weight': 0.5,
        'uncertainty_deg': 20.0,
        'uncertainty_growth': 1.0,
    }


def candidate_key(config):
    return tuple(config[key] for key in PARAMETER_KEYS)


def one_factor_candidates():
    """The original 16-case one-factor-at-a-time sweep."""
    base = base_candidate()
    rows = [base]
    sweeps = {
        'top_k': [4, 6, 10, 12],
        'velocity_window': [2, 5],
        'horizon_scale': [0.5, 1.5, 2.0],
        'acceleration_weight': [0.0, 1.0],
        'uncertainty_deg': [0.0, 40.0],
        'uncertainty_growth': [0.0, 2.0],
    }
    for key, values in sweeps.items():
        for value in values:
            candidate = dict(base)
            candidate[key] = value
            rows.append(candidate)
    return rows


def _normalized_candidate(config):
    coordinates = []
    for key in PARAMETER_KEYS:
        values = PARAMETER_SPACE[key]
        denominator = max(len(values) - 1, 1)
        coordinates.append(values.index(config[key]) / denominator)
    return tuple(coordinates)


def _distance(left, right):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def interaction_candidates(count=32):
    """Deterministic space-filling multi-parameter candidates.

    Candidates differ from the center in at least two parameters. A greedy
    maximin rule spreads them across the finite search space while retaining
    deterministic, reproducible ordering.
    """
    if count <= 0:
        return []
    base = base_candidate()
    excluded = {candidate_key(row) for row in one_factor_candidates()}
    pool = []
    for values in itertools.product(*(PARAMETER_SPACE[key] for key in PARAMETER_KEYS)):
        config = dict(zip(PARAMETER_KEYS, values))
        if candidate_key(config) in excluded:
            continue
        changed = sum(config[key] != base[key] for key in PARAMETER_KEYS)
        if changed >= 2:
            pool.append(config)
    if count > len(pool):
        raise ValueError(f'requested {count} interactions from a pool of {len(pool)}')

    anchors = [_normalized_candidate(row) for row in one_factor_candidates()]
    level_counts = {
        key: {value: 0 for value in PARAMETER_SPACE[key]}
        for key in PARAMETER_KEYS
    }
    selected = []
    while len(selected) < count:
        scored = []
        for config in pool:
            point = _normalized_candidate(config)
            min_distance = min(_distance(point, anchor) for anchor in anchors)
            usage_penalty = sum(
                level_counts[key][config[key]] for key in PARAMETER_KEYS
            )
            scored.append((usage_penalty, min_distance, candidate_key(config), config, point))
        _, _, _, chosen, chosen_point = max(
            scored, key=lambda item: (-item[0], item[1], item[2])
        )
        selected.append(chosen)
        anchors.append(chosen_point)
        for key in PARAMETER_KEYS:
            level_counts[key][chosen[key]] += 1
        pool.remove(chosen)
    return selected


def candidates(interaction_count=32):
    rows = one_factor_candidates() + interaction_candidates(interaction_count)
    if len({candidate_key(row) for row in rows}) != len(rows):
        raise RuntimeError('kinematic selector search contains duplicate candidates')
    return rows


def aggregate_result(directory):
    result_files = sorted(
        path for path in directory.glob('*_results.csv')
        if '_partial_' not in path.name and '_per_sample_' not in path.name
    )
    if not result_files:
        raise FileNotFoundError(f'aggregate results CSV absent in {directory}')
    with result_files[-1].open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    aggregate = next(
        row for row in rows
        if int(float(row['video'])) == -1 and int(float(row['user'])) == -1
    )
    with (directory / 'latency.json').open(encoding='utf-8') as handle:
        latency = json.load(handle)
    return {
        'mae': float(aggregate['mae']),
        'rmse': float(aggregate['rmse']),
        'latency_mean_ms': float(latency['mean_s']) * 1000.0,
    }


def command(args, config, output_dir, split, limit):
    cmd = [
        sys.executable, 'run_plm.py', '--test',
        '--train-dataset', 'Jin2022', '--test-dataset', 'Jin2022',
        '--evaluation-split', split,
        '--plm-type', 'llama', '--plm-size', 'base',
        '--device', args.device, '--device-out', args.device,
        '--fp16', '--rank', '32', '--use-adalora',
        '--adalora-allocator', 'nbs',
        '--adalora-rank-config', 'configs/adalora_rank_config_llama7b_min2_max32.json',
        '--adalora-rank-budget', '512', '--adalora-ema-beta', '0.9',
        '--adalora-shadow-update-policy', 'legacy',
        '--adalora-allocation-interval', '10', '--experiment-tag', 'nbs_v19',
        '--model-path', str(args.nbs_checkpoint), '--evaluation-tag', 'best_ar',
        '--nbs-inference-mode', 'original',
        '--multimodal-mode', 'patch-selection',
        '--patch-selector-type', 'kinematic',
        '--patch-top-k', str(config['top_k']),
        '--kinematic-velocity-window', str(config['velocity_window']),
        '--kinematic-horizon-scale', str(config['horizon_scale']),
        '--kinematic-acceleration-weight', str(config['acceleration_weight']),
        '--kinematic-uncertainty-deg', str(config['uncertainty_deg']),
        '--kinematic-uncertainty-growth', str(config['uncertainty_growth']),
        '--multimodal-projector-checkpoint', str(args.projector_checkpoint),
        '--epochs', '4', '--bs', '1', '--grad-accum-steps', '32',
        '--lr', '0.0002', '--seed', '1', '--lora-seed', '1',
        '--data-seed', '1', '--measure-inference-latency',
        '--latency-warmup-steps', '5',
        '--latency-output-path', str(output_dir / 'latency.json'),
        '--results-output-dir', str(output_dir), '--inference-tag', 'selector',
    ]
    if limit:
        cmd.extend([
            '--limit-test-samples', str(limit),
            '--limit-test-samples-random',
        ])
    return cmd


def run_case(args, config, output_dir, split, limit):
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / 'metrics.json'
    if args.resume and metrics_path.is_file():
        with metrics_path.open(encoding='utf-8') as handle:
            return json.load(handle)
    cmd = command(args, config, output_dir, split, limit)
    with (output_dir / 'command.json').open('w', encoding='utf-8') as handle:
        json.dump(cmd, handle, indent=2)
    with (output_dir / 'run.log').open('w', encoding='utf-8') as log:
        subprocess.run(
            cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
            check=True,
        )
    metrics = aggregate_result(output_dir)
    with metrics_path.open('w', encoding='utf-8') as handle:
        json.dump(metrics, handle, indent=2)
    return metrics


def write_rows(path, rows):
    fieldnames = sorted({key for row in rows for key in row})
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_stage(args, configs, stage, split, limit):
    rows = []
    for index, config in enumerate(configs, start=1):
        case_dir = args.output_dir / stage / f'candidate_{index:02d}'
        row = {'candidate': index, 'stage': stage, **config, 'status': 'failed'}
        try:
            row.update(run_case(args, config, case_dir, split, limit))
            row['status'] = 'complete'
        except Exception as exc:
            row['error'] = f'{type(exc).__name__}: {exc}'
        rows.append(row)
        print(
            f"[{stage} {index}/{len(configs)}] {row['status']} "
            f"MAE={row.get('mae')} config={config}", flush=True,
        )
        write_rows(args.output_dir / f'{stage}_results.csv', rows)
    completed = [row for row in rows if row['status'] == 'complete']
    if not completed:
        raise RuntimeError(f'every candidate failed during {stage}')
    return sorted(completed, key=lambda row: (row['mae'], row['latency_mean_ms']))


def row_config(row):
    return {key: row[key] for key in PARAMETER_KEYS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nbs-checkpoint', type=Path, default=DEFAULT_NBS)
    parser.add_argument(
        '--projector-checkpoint', type=Path,
        default=Path('patch_selection_delivery/model/vp_3condition_best_model'),
    )
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--validation-samples', type=int, default=256)
    parser.add_argument('--interaction-candidates', type=int, default=32)
    parser.add_argument('--refinement-candidates', type=int, default=12)
    parser.add_argument('--refinement-samples', type=int, default=512)
    parser.add_argument('--full-validation-candidates', type=int, default=5)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    args.nbs_checkpoint = (REPO_ROOT / args.nbs_checkpoint).resolve()
    args.projector_checkpoint = (REPO_ROOT / args.projector_checkpoint).resolve()
    args.output_dir = (REPO_ROOT / args.output_dir).resolve()
    if args.validation_samples <= 0:
        parser.error('--validation-samples must be positive')
    if args.refinement_samples <= args.validation_samples:
        parser.error('--refinement-samples must exceed --validation-samples')
    if args.interaction_candidates < 0:
        parser.error('--interaction-candidates must be non-negative')
    search_candidates = candidates(args.interaction_candidates)
    if not 0 < args.refinement_candidates <= len(search_candidates):
        parser.error('--refinement-candidates must fit within the screening set')
    if not 0 < args.full_validation_candidates <= args.refinement_candidates:
        parser.error('--full-validation-candidates must fit within refinement')
    if args.dry_run:
        print(json.dumps({
            'screening_count': len(search_candidates),
            'one_factor_count': len(one_factor_candidates()),
            'interaction_count': args.interaction_candidates,
            'refinement_count': args.refinement_candidates,
            'full_validation_count': args.full_validation_candidates,
            'candidates': search_candidates,
        }, indent=2))
        return
    for required_data in (
            REPO_ROOT / 'viewport_prediction/data/viewports/Jin2022/video1/5Hz/simple_5Hz_user50.csv',
            REPO_ROOT / 'viewport_prediction/data/images/Jin2022_images'):
        if not required_data.exists():
            parser.error(f'missing VP evaluation data: {required_data}')
    for required in (
            args.nbs_checkpoint / 'adapter_model.bin',
            args.nbs_checkpoint / 'modules_except_plm.bin',
            args.nbs_checkpoint / 'nash_rank_allocator.pt'):
        if not required.is_file():
            parser.error(f'missing NBS checkpoint file: {required}')
    projector_file = (
        args.projector_checkpoint / 'modules_except_plm.bin'
        if args.projector_checkpoint.is_dir() else args.projector_checkpoint
    )
    if not projector_file.is_file():
        parser.error(f'missing projector checkpoint: {projector_file}')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    screening = evaluate_stage(
        args, search_candidates, 'screening', 'valid', args.validation_samples
    )
    refinement_configs = [
        row_config(row) for row in screening[:args.refinement_candidates]
    ]
    refinement = evaluate_stage(
        args, refinement_configs, 'refinement', 'valid', args.refinement_samples
    )
    full_configs = [
        row_config(row) for row in refinement[:args.full_validation_candidates]
    ]
    full_validation = evaluate_stage(
        args, full_configs, 'full_validation', 'valid', None
    )
    best = full_validation[0]
    best_config = row_config(best)
    final_metrics = run_case(
        args, best_config, args.output_dir / 'best_test', 'test', None
    )
    summary = {
        'selection_split': 'valid',
        'selection_metric': 'rotation_aware_mae_degrees',
        'screening_candidate_count': len(search_candidates),
        'screening_samples_per_candidate': args.validation_samples,
        'refinement_candidate_count': args.refinement_candidates,
        'refinement_samples_per_candidate': args.refinement_samples,
        'full_validation_candidate_count': args.full_validation_candidates,
        'best_candidate': best,
        'best_test_metrics': final_metrics,
        'test_set_used_for_search': False,
    }
    with (args.output_dir / 'best_configuration.json').open(
            'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)
    print('Best kinematic selector:', json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
