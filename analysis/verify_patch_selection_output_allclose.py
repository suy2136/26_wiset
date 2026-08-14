"""
Step 2 of the crop-optimization verification plan: confirm that the FULL output of
_get_multimodal_information_patch_selection() (crop -> ViT -> embed_multimodal) is
numerically unchanged by the crop_patches_at() optimization.

Approach: build one real Pipeline (real vit_b_16, real embed_multimodal, a real trained
PatchSelectionModule) and compute two outputs for the same input:
  - "new": pipeline._get_multimodal_information_patch_selection(...) as it exists now
    (selective crop via crop_patches_at())
  - "old": the pre-optimization code path, reconstructed inline here (crop_patches() the
    FULL grid, then index into it) -- i.e. exactly what _load_frame_patches() used to do
    before this change.
Compare with torch.allclose(atol=1e-2), same tolerance used for the earlier KV-cache
verification (fp16 forward -> nonzero float rounding is expected).

Run inside the vp_netllm conda env:
    conda activate vp_netllm && python analysis/verify_patch_selection_output_allclose.py
"""
import os
import sys

import torch
import torch.nn as nn
import torchvision

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import cfg
from models.patch_selection import PatchSelectionModule, crop_patches, vit_features_for_patches
from models.pipeline import Pipeline

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT_PATH = '/workspace/data/models/patch_selection/best_patch_selection.pth'


class DummyPLM:
    """Stand-in for the LLM -- _get_multimodal_information_patch_selection() never
    touches self.plm, only Pipeline.__init__ needs a .networking_head to build
    modules_except_plm."""
    def __init__(self):
        self.networking_head = nn.Identity()


def old_get_multimodal_information_patch_selection(pipeline, video_user_position, history_viewports):
    """Exact reconstruction of the pre-optimization code path (crop the full 16-patch
    grid via crop_patches(), THEN index into it) for comparison."""
    video_index, image_index = pipeline._resolve_frame_index(video_user_position)
    with torch.no_grad():
        logits = pipeline.patch_selection_module(history_viewports.to(pipeline.device))
    mask = pipeline.patch_selection_module.select_patches(
        logits, top_k=pipeline.patch_top_k, threshold=pipeline.patch_threshold)
    indices = mask[0].nonzero(as_tuple=True)[0].tolist()
    if len(indices) == 0:
        indices = [logits[0].argmax().item()]

    ext = cfg.dataset_image_ext[pipeline.dataset]
    from PIL import Image
    image_path = os.path.join(cfg.dataset_images[pipeline.dataset], f'video{video_index}_images',
                                f'{image_index}.{ext}')
    image = Image.open(image_path).convert('RGB')
    image_tensor = pipeline.raw_image_transform(image)
    patches = crop_patches(image_tensor, pipeline.grid_rows, pipeline.grid_cols)  # OLD: always all 16
    features = vit_features_for_patches(patches, indices, pipeline._vit_feature_fn, device=pipeline.device)
    mapped_tensor = pipeline.embed_multimodal(features)
    return mapped_tensor.unsqueeze(0), indices


def main():
    print(f'device: {DEVICE}')
    vit_model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT).to(DEVICE)

    patch_selection_module = PatchSelectionModule(grid_rows=4, grid_cols=4).to(DEVICE)
    patch_selection_module.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    patch_selection_module.eval()

    pipeline = Pipeline(
        plm=DummyPLM(), fut_window=20, device=DEVICE, embed_size=1024, frequency=5,
        multimodal_mode='patch-selection', dataset='Jin2022',
        patch_selection_module=patch_selection_module, vit_model=vit_model,
        patch_top_k=None, patch_threshold=None,  # match the default (threshold=0.5) used throughout tasks 3/4
    ).to(DEVICE)
    pipeline.eval()

    # a handful of real (video, user, timestep) samples spanning different videos/positions
    test_cases = [
        (4, 1, 30), (4, 1, 300), (8, 1, 750), (14, 1, 1200), (18, 1, 60),
    ]
    torch.manual_seed(0)
    all_ok = True
    for video, user, timestep in test_cases:
        video_user_position = torch.tensor([video, user, timestep])
        history_viewports = torch.randn(1, 10, 3) * 30  # plausible-scale roll/pitch/yaw history

        with torch.no_grad():
            new_out = pipeline._get_multimodal_information_patch_selection(video_user_position, history_viewports)
            old_out, indices = old_get_multimodal_information_patch_selection(pipeline, video_user_position, history_viewports)

        ok = new_out.shape == old_out.shape and torch.allclose(new_out, old_out, atol=1e-2)
        all_ok &= ok
        max_diff = (new_out - old_out).abs().max().item()
        print(f'video={video} user={user} t={timestep}: indices={indices} '
              f'shape={tuple(new_out.shape)} max_abs_diff={max_diff:.6f} allclose(atol=1e-2)={ok}')

    print(f'\n{"ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"}')
    if not all_ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
