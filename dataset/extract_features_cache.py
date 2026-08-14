"""
One-time offline preprocessing: extract ViT CLS-token features for every Jin2022 video
frame and cache them to disk, in the exact layout models/pipeline.py.get_multimodal_information()
expects for the `baseline` multimodal mode (single CLS token per frame, offline cache).

This replaces dataset/extract_features.py's Wu2017-specific, hardcoded-absolute-path script:
- source images: cfg.dataset_images['Jin2022']/video{N}_images/{k}.jpg (real path/extension)
- target cache:  cfg.dataset_image_features['Jin2022']/video{N}_images/feature_dict{b}.pth
- frame counts are read from disk (not assumed to be 1500/1800), and the true counts are
  also written to cfg.dataset_frame_count_manifest['Jin2022'] for utils/frame_utils.py to use.

Run this once on the GPU server before using --multimodal-mode baseline (or all-patch /
patch-selection, which also read the same manifest for frame-count clamping).

Usage: python -m dataset.extract_features_cache
"""
import json
import os
import sys

import torch
import torchvision
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import cfg
from dataset.extract_features import extract_vit_features

JIN2022_VIDEO_NUM = 27
SAVE_EVERY = 100  # matches the batching convention pipeline.py expects (feature_dict{n//100}.pth)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def count_frames(video_dir, ext):
    files = [f for f in os.listdir(video_dir) if f.endswith(f'.{ext}')]
    return len(files)


def extract_one_video(video_index, model, device, images_root, features_root, ext):
    video_dir = os.path.join(images_root, f'video{video_index}_images')
    count = count_frames(video_dir, ext)
    if count == 0:
        print(f'\033[33mWarning:\033[0m no .{ext} frames found for video{video_index} at {video_dir}, skipping.')
        return 0

    target_dir = os.path.join(features_root, f'video{video_index}_images')
    os.makedirs(target_dir, exist_ok=True)

    tensor_dict = {}
    for n in range(1, count + 1):
        img_path = os.path.join(video_dir, f'{n}.{ext}')
        img = Image.open(img_path).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feature = extract_vit_features(img_tensor, model=model).cpu()
        tensor_dict[f'{n}'] = feature

        if n % SAVE_EVERY == 0:
            torch.save(tensor_dict, os.path.join(target_dir, f'feature_dict{n // SAVE_EVERY}.pth'))
            tensor_dict = {}
            print(f'video{video_index}: cached {n}/{count} frames', flush=True)

    if tensor_dict:  # flush the remainder (count is not a multiple of SAVE_EVERY)
        torch.save(tensor_dict, os.path.join(target_dir, f'feature_dict{(count // SAVE_EVERY) + 1}.pth'))
        print(f'video{video_index}: cached {count}/{count} frames', flush=True)

    return count


def main():
    dataset = 'Jin2022'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    images_root = cfg.dataset_images[dataset]
    features_root = cfg.dataset_image_features[dataset]
    ext = cfg.dataset_image_ext[dataset]
    manifest_path = cfg.dataset_frame_count_manifest[dataset]

    os.makedirs(features_root, exist_ok=True)

    print(f'Loading ViT-B/16 (pretrained) on {device}...')
    model = torchvision.models.vit_b_16(pretrained=True).to(device)
    model.eval()

    frame_counts = {}
    for video_index in range(1, JIN2022_VIDEO_NUM + 1):
        print(f'--- video{video_index} ---')
        count = extract_one_video(video_index, model, device, images_root, features_root, ext)
        frame_counts[video_index] = count

    with open(manifest_path, 'w') as f:
        json.dump(frame_counts, f, indent=2)
    print(f'Wrote frame count manifest to {manifest_path}')
    print('Done. Frame counts:', frame_counts)


if __name__ == '__main__':
    main()
