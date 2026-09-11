"""Precompute per-patch ViT features for low-latency VP evaluation.

Each output contains one FP16 tensor shaped ``[frames, 16, 768]``.  It uses
the same raw-frame cropping and frozen torchvision ViT path as the existing
online patch-selection implementation, but pays that cost once offline.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch
import torchvision
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import cfg
from dataset.extract_features import extract_vit_features
from models.patch_selection import crop_patches, vit_features_for_patches


def parser():
    result = argparse.ArgumentParser()
    result.add_argument("--dataset", default="Jin2022", choices=["Jin2022"])
    result.add_argument("--output-dir", type=Path, default=None)
    result.add_argument("--videos", type=int, nargs="+", default=list(range(1, 28)))
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--resume", action="store_true")
    return result


def numbered_frames(video_dir, extension):
    frames = []
    for path in video_dir.glob(f"*.{extension}"):
        try:
            frames.append((int(path.stem), path))
        except ValueError:
            continue
    frames.sort()
    if not frames:
        raise FileNotFoundError(f"no numbered .{extension} frames in {video_dir}")
    expected = list(range(1, len(frames) + 1))
    actual = [number for number, _ in frames]
    if actual != expected:
        raise ValueError(
            f"frames in {video_dir} must be contiguous and one-based; "
            f"found {actual[:3]}...{actual[-3:]}"
        )
    return frames


def atomic_torch_save(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def extract_video(video, frames, model, device, grid):
    transform = transforms.ToTensor()
    output = []
    for offset, (_, path) in enumerate(frames, 1):
        with Image.open(path) as image:
            image_tensor = transform(image.convert("RGB"))
        patches = crop_patches(image_tensor, *grid)
        features = vit_features_for_patches(
            patches, range(patches.shape[0]),
            lambda batch: extract_vit_features(batch, model=model),
            device=device,
        )
        output.append(features.cpu().to(torch.float16))
        if offset % 100 == 0 or offset == len(frames):
            print(f"video{video}: {offset}/{len(frames)}", flush=True)
    result = torch.stack(output)
    if not torch.isfinite(result).all():
        raise RuntimeError(f"video{video} produced non-finite ViT features")
    return result


def validate_existing(path, frame_count, patch_count):
    payload = torch.load(path, map_location="cpu")
    features = payload.get("features") if isinstance(payload, dict) else payload
    return (
        torch.is_tensor(features)
        and tuple(features.shape) == (frame_count, patch_count, 768)
        and features.dtype == torch.float16
        and bool(torch.isfinite(features).all())
    )


def main():
    args = parser().parse_args()
    output_dir = args.output_dir or Path(cfg.dataset_patch_features[args.dataset])
    output_dir.mkdir(parents=True, exist_ok=True)
    images_root = Path(cfg.dataset_images[args.dataset])
    extension = cfg.dataset_image_ext[args.dataset]
    grid = tuple(cfg.default_patch_grid)
    patch_count = grid[0] * grid[1]

    print(f"Loading frozen ViT-B/16 on {args.device}", flush=True)
    model = torchvision.models.vit_b_16(pretrained=True).to(args.device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    manifest = {
        "dataset": args.dataset,
        "grid": list(grid),
        "feature_dim": 768,
        "dtype": "float16",
        "videos": {},
    }
    for video in args.videos:
        video_dir = images_root / f"video{video}_images"
        frames = numbered_frames(video_dir, extension)
        target = output_dir / f"video{video}_patch_features.pt"
        if args.resume and target.is_file() and validate_existing(
                target, len(frames), patch_count):
            print(f"video{video}: valid cache exists; skipping", flush=True)
        else:
            features = extract_video(video, frames, model, args.device, grid)
            atomic_torch_save({
                "features": features,
                "video": video,
                "grid": list(grid),
                "frame_count": len(frames),
                "feature_dim": 768,
                "dtype": "float16",
            }, target)
        manifest["videos"][str(video)] = {
            "frames": len(frames), "file": target.name,
        }

    manifest_path = output_dir / "manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temporary, manifest_path)
    print(f"Saved patch cache manifest: {manifest_path}")


if __name__ == "__main__":
    main()
