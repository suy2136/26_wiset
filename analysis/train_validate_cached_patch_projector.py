"""One-epoch frozen-NBS projector fine-tuning from cached VP patch features."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_VIDEOS = (1, 5, 9, 2, 6, 11, 15, 16, 13, 17, 21, 22, 26, 19, 23)
VALID_VIDEOS = (3, 7, 12, 10, 20, 27)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument('--nbs-checkpoint', type=Path, required=True)
    result.add_argument('--projector-checkpoint', type=Path, required=True)
    result.add_argument('--cache-dir', type=Path, required=True)
    result.add_argument('--output-dir', type=Path, required=True)
    result.add_argument('--device', default='cuda:0')
    result.add_argument('--rank-budget', type=int, default=512)
    result.add_argument('--physical-rank', type=int, default=32)
    result.add_argument('--lr', type=float, default=5e-5)
    result.add_argument(
        '--policy',
        choices=('k1', 'adaptive', 'k2', 'cross',
                 'gated-k1', 'gated-adaptive'),
        default='gated-k1',
        help=(
            'Cached patch policy used for both projector training and '
            'validation. Use k1 to update the projector on every sample.'
        ),
    )
    result.add_argument('--validation-samples', type=int, default=128)
    result.add_argument('--log-every', type=int, default=100)
    result.add_argument(
        '--cache-device', choices=('cpu', 'model'), default='cpu',
        help='Keep the multi-video cache on CPU by default to avoid GPU OOM.',
    )
    result.add_argument('--limit-train-samples', type=int)
    result.add_argument('--limit-valid-samples', type=int)
    result.add_argument('--dry-run', action='store_true')
    return result


def absolute(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def require_checkpoint(path: Path) -> None:
    required = ('adapter_config.json', 'modules_except_plm.bin',
                'nash_rank_allocator.pt')
    missing = [name for name in required if not (path / name).is_file()]
    if not ((path / 'adapter_model.bin').is_file()
            or (path / 'adapter_model.safetensors').is_file()):
        missing.append('adapter_model.bin or adapter_model.safetensors')
    if missing:
        raise FileNotFoundError(
            f'incomplete NBS checkpoint {path}: {", ".join(missing)}'
        )


def build_command(args) -> list[str]:
    output_dir = absolute(args.output_dir)
    command = [
        sys.executable, 'run_plm.py', '--adapt', '--resume',
        '--resume-path', str(absolute(args.nbs_checkpoint)),
        '--train-dataset', 'Jin2022', '--test-dataset', 'Jin2022',
        '--plm-type', 'llama', '--plm-size', 'base',
        '--device', args.device, '--device-out', args.device,
        '--fp16', '--rank', str(args.physical_rank), '--use-adalora',
        '--adalora-allocator', 'nbs',
        '--adalora-rank-config',
        'configs/adalora_rank_config_llama7b_min2_max32.json',
        '--adalora-rank-budget', str(args.rank_budget),
        '--experiment-tag', 'nbs_v19',
        '--multimodal-mode', 'cached-patch-selection',
        '--cached-patch-policy', args.policy,
        '--cached-patch-motion-threshold-deg', '6',
        '--cached-patch-max-skip-calls', '0',
        '--cached-patch-features-dir', str(absolute(args.cache_dir)),
        '--cached-patch-cache-device', args.cache_device,
        '--multimodal-projector-checkpoint',
        str(absolute(args.projector_checkpoint)),
        '--train-multimodal-projector-only',
        '--multimodal-projector-output-dir', str(output_dir),
        '--multimodal-projector-validation-samples',
        str(args.validation_samples),
        '--multimodal-projector-log-every', str(args.log_every),
        '--epochs', '1', '--bs', '1', '--grad-accum-steps', '1',
        '--lr', str(args.lr), '--weight-decay', '0.0001',
        '--seed', '1', '--lora-seed', '1', '--data-seed', '1',
    ]
    if args.limit_train_samples is not None:
        command.extend(['--limit-train-samples', str(args.limit_train_samples)])
    if args.limit_valid_samples is not None:
        command.extend(['--limit-valid-samples', str(args.limit_valid_samples)])
    return command


def run_streaming(command: list[str], log_path: Path) -> None:
    environment = os.environ.copy()
    environment['PYTHONUNBUFFERED'] = '1'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(
            command, cwd=REPO_ROOT, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
            encoding='utf-8', errors='replace', env=environment,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def main() -> None:
    args = parser().parse_args()
    args.nbs_checkpoint = absolute(args.nbs_checkpoint)
    args.projector_checkpoint = absolute(args.projector_checkpoint)
    args.cache_dir = absolute(args.cache_dir)
    args.output_dir = absolute(args.output_dir)
    if not args.dry_run:
        require_checkpoint(args.nbs_checkpoint)
        if not args.projector_checkpoint.exists():
            raise FileNotFoundError(args.projector_checkpoint)
        if not args.cache_dir.is_dir():
            raise FileNotFoundError(args.cache_dir)
        required_videos = TRAIN_VIDEOS + VALID_VIDEOS
        missing_cache = [
            video for video in required_videos
            if not (args.cache_dir /
                    f'video{video}_patch_features.pt').is_file()
        ]
        if missing_cache:
            videos = ' '.join(str(video) for video in missing_cache)
            raise FileNotFoundError(
                'cached patch training/validation features are incomplete; '
                f'missing videos: {videos}. Generate them with: '
                'python dataset/extract_patch_features_cache.py '
                f'--videos {videos} --output-dir {args.cache_dir} '
                '--device cuda:0 --resume'
            )
    command = build_command(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'command.json').write_text(
        json.dumps(command, indent=2), encoding='utf-8'
    )
    print(' '.join(command), flush=True)
    if args.dry_run:
        return
    run_streaming(command, args.output_dir / 'run.log')
    required_outputs = (
        'best_multimodal_projector.pth',
        'latest_multimodal_projector.pth',
        'projector_training_history.csv',
        'projector_training_manifest.json',
    )
    missing = [name for name in required_outputs
               if not (args.output_dir / name).is_file()]
    if missing:
        raise RuntimeError(f'projector pipeline missing outputs: {missing}')
    print('Validated projector outputs:', args.output_dir, flush=True)


if __name__ == '__main__':
    main()
