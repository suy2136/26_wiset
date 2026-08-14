"""
Verification for the crop_patches_at() / _load_frame_patches(indices=...) optimization
(see analysis/PATCH_SELECTION_ANALYSIS.md and IMPLEMENTATION_NOTES.md section 5).

Two checks:
  1. _load_frame_patches(indices=None) on a REAL frame still returns exactly what the
     OLD code path did: crop_patches(full image) -- i.e. the all-patch path is provably
     untouched.
  2. _load_frame_patches(indices=[...]) on a REAL frame exactly matches
     crop_patches(full image)[indices] -- i.e. the new selective-crop path produces
     numerically identical patches to what the old "crop all 16, then index" path would
     have produced for the same indices, INCLUDING across repeated calls with different
     index subsets for the same cached frame (the caching-correctness concern).

Run inside the vp_netllm conda env:
    conda activate vp_netllm && python analysis/verify_patch_crop_optimization.py
"""
import os
import sys
import types

import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import cfg
from models.patch_selection import crop_patches
from models.pipeline import Pipeline

SAMPLE_FRAMES = [
    ('Jin2022', 4, 1), ('Jin2022', 4, 750), ('Jin2022', 8, 1500), ('Jin2022', 14, 300),
]


def make_fake_self(dataset, grid_rows=4, grid_cols=4):
    fake = types.SimpleNamespace()
    fake.dataset = dataset
    fake.grid_rows = grid_rows
    fake.grid_cols = grid_cols
    fake.raw_image_transform = transforms.ToTensor()
    fake._patch_image_cache = {}
    return fake


def load_reference_patches(dataset, video_index, image_index, grid_rows, grid_cols):
    ext = cfg.dataset_image_ext[dataset]
    image_path = os.path.join(cfg.dataset_images[dataset], f'video{video_index}_images', f'{image_index}.{ext}')
    image = Image.open(image_path).convert('RGB')
    image_tensor = transforms.ToTensor()(image)
    return crop_patches(image_tensor, grid_rows, grid_cols)


def main():
    all_ok = True

    print('=== check 1: _load_frame_patches(indices=None) matches old crop_patches(full) ===')
    for dataset, video_index, image_index in SAMPLE_FRAMES:
        fake = make_fake_self(dataset)
        got = Pipeline._load_frame_patches(fake, video_index, image_index)  # indices=None default
        ref = load_reference_patches(dataset, video_index, image_index, fake.grid_rows, fake.grid_cols)
        ok = torch.equal(got, ref)
        all_ok &= ok
        print(f'  video{video_index}/{image_index}: shape={tuple(got.shape)} exact_match={ok}')

    print('\n=== check 2: _load_frame_patches(indices=[...]) matches crop_patches(full)[indices] ===')
    index_subsets = [
        [6, 7, 10, 11],           # the "hot" horizon patches from task 3
        [0, 1, 12, 13],           # the "cold" pole patches
        [3, 9, 2, 14, 5],         # arbitrary order/subset, not sorted
        list(range(16)),          # full grid via the selective path
    ]
    for dataset, video_index, image_index in SAMPLE_FRAMES:
        fake = make_fake_self(dataset)  # fresh cache per frame
        ref = load_reference_patches(dataset, video_index, image_index, fake.grid_rows, fake.grid_cols)
        for indices in index_subsets:
            got = Pipeline._load_frame_patches(fake, video_index, image_index, indices=indices)
            expected = ref[indices]
            ok = torch.equal(got, expected)
            all_ok &= ok
            print(f'  video{video_index}/{image_index} indices={indices}: shape={tuple(got.shape)} exact_match={ok}')
        # cache should now hold exactly one entry (the decoded image) for this frame,
        # reused correctly across all the differing index requests above
        assert len(fake._patch_image_cache) == 1, 'expected exactly one cached decoded image per frame'

    print('\n=== check 3: interleaved calls with different index subsets on the SAME fake-self cache ===')
    dataset, video_index, image_index = SAMPLE_FRAMES[0]
    fake = make_fake_self(dataset)
    ref = load_reference_patches(dataset, video_index, image_index, fake.grid_rows, fake.grid_cols)
    call_sequence = [[6, 7], [0, 1, 12, 13], [6, 7, 10, 11], list(range(16)), [15]]
    for indices in call_sequence:
        got = Pipeline._load_frame_patches(fake, video_index, image_index, indices=indices)
        ok = torch.equal(got, ref[indices])
        all_ok &= ok
        print(f'  indices={indices}: exact_match={ok}')

    print(f'\n{"ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"}')
    if not all_ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
