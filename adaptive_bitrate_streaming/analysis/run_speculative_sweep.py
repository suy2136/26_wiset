"""Compare ordinary ABR inference with robust-MPC draft horizons K=1..4."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


ABR_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ABR_ROOT / 'artifacts' / 'results'


def configurations(draft_steps):
    yield 0
    yield from draft_steps


def command_for(steps, mode, tolerance, forwarded):
    command = [sys.executable, 'run_plm.py', '--test', *forwarded]
    command.extend(['--speculative-draft-steps', str(steps)])
    if steps > 0:
        command.extend(['--speculative-verification-mode', mode])
        command.extend(['--speculative-buffer-tolerance', str(tolerance)])
    return command


def newest_metrics(steps, mode, tolerance):
    tag = (
        'speculative_none' if steps == 0
        else f'speculative_mpc_k{steps}_{mode}_btol{tolerance}'
    )
    candidates = [
        path for path in RESULTS_ROOT.rglob('selector_metrics.json')
        if tag in path.parts
    ]
    if not candidates:
        raise FileNotFoundError(f'no metrics produced for {tag}')
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--draft-steps', type=int, nargs='+', default=[1, 2, 3, 4])
    parser.add_argument('--verification-mode', choices=('greedy', 'sample'), default='sample')
    parser.add_argument('--buffer-tolerance', type=float, default=1.0)
    parser.add_argument('--output-csv', default='artifacts/results/speculative_sweep.csv')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('forwarded', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    forwarded = args.forwarded[1:] if args.forwarded[:1] == ['--'] else args.forwarded
    forbidden = {
        '--speculative-draft-steps', '--speculative-verification-mode',
        '--speculative-buffer-tolerance',
    }
    if forbidden.intersection(forwarded):
        parser.error('speculative options are controlled by the sweep script')
    if any(steps <= 0 for steps in args.draft_steps):
        parser.error('--draft-steps values must be positive')
    if args.buffer_tolerance < 0:
        parser.error('--buffer-tolerance must be non-negative')

    rows = []
    for steps in configurations(args.draft_steps):
        command = command_for(
            steps, args.verification_mode, args.buffer_tolerance, forwarded
        )
        print(' '.join(command))
        if args.dry_run:
            continue
        subprocess.run(command, cwd=ABR_ROOT, check=True)
        metrics_path = newest_metrics(
            steps, args.verification_mode, args.buffer_tolerance
        )
        with metrics_path.open() as f:
            row = json.load(f)
        row['metrics_path'] = str(metrics_path)
        rows.append(row)

    if args.dry_run:
        return
    output_path = ABR_ROOT / args.output_csv
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        'speculative_draft_steps', 'mean_reward',
        'inference_latency_mean_ms', 'inference_latency_p95_ms',
        'target_plm_calls', 'llm_call_reduction_ratio', 'acceptance_rate',
        'draft_attempts', 'drafted_actions', 'accepted_actions',
        'corrected_actions', 'queued_actions_served',
        'state_mismatch_fallbacks', 'draft_generation_failures',
        'token_reduction_ratio', 'time', 'metrics_path',
    ]
    with output_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {output_path}')


if __name__ == '__main__':
    main()
