"""
Step 3 of the crop-optimization verification plan: re-measure patch-selection and
all-patch latency AFTER the crop_patches_at() optimization
(models/patch_selection.py, models/pipeline.py:290-337), under the same conditions as
the earlier KV-cache benchmark in IMPLEMENTATION_NOTES.md section 3 (RTX 5090,
llama-7b fp16, batch=1, warmup=3, iters=10, utils.latency_utils.measure_inference_latency).

Also measures all-patch mode as a regression check -- its code path takes
indices=None through _load_frame_patches() and is untouched by this change, so its
latency should be within noise of the pre-optimization number.

Run inside the vp_netllm conda env:
    conda activate vp_netllm && python analysis/benchmark_patch_crop_optimization.py
"""
import os
import sys

import torch
import torchvision

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import cfg
from dataset.load_dataset import create_dataset
from models.networking_head import NetworkingHead
from models.patch_selection import PatchSelectionModule
from models.pipeline import Pipeline
from utils.latency_utils import measure_inference_latency, format_latency_report
from utils.normalize import normalize_data
from utils.plms_utils import load_plm

DEVICE = 'cuda'
CKPT_PATH = '/workspace/data/models/patch_selection/best_patch_selection.pth'


def build_pipeline(multimodal_mode, plm, networking_head, vit_model, patch_selection_module):
    plm.set_networking_head(networking_head)
    return Pipeline(
        plm, fut_window=20, device=DEVICE, embed_size=4096, frequency=cfg.default_dataset_frequency,
        multimodal_mode=multimodal_mode, dataset='Jin2022',
        patch_selection_module=patch_selection_module, vit_model=vit_model,
        patch_top_k=None, patch_threshold=None,
    ).to(DEVICE)


def main():
    print(f'device: {DEVICE}, gpu: {torch.cuda.get_device_name(0)}')

    print('loading llama-7b (fp16)...')
    plm, tokenizer, _ = load_plm(
        'llama', os.path.join(cfg.plms_dir, 'llama', 'base'),
        device_input_side=DEVICE, device_output_side=DEVICE, torch_dtype=torch.float16)
    plm = plm.to(DEVICE)
    for p in plm.parameters():
        p.requires_grad_(False)

    vit_model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT).to(DEVICE)

    patch_selection_module = PatchSelectionModule(grid_rows=4, grid_cols=4).to(DEVICE)
    patch_selection_module.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    patch_selection_module.eval()

    (test_ds,) = create_dataset('Jin2022', include=('test',))
    history, future, video_user_info = test_ds[0]
    history = normalize_data(torch.tensor(history).unsqueeze(0), 'Jin2022').to(DEVICE)
    future = normalize_data(torch.tensor(future).unsqueeze(0), 'Jin2022').to(DEVICE)
    video_user_position = [torch.tensor([v]) for v in video_user_info]  # matches DataLoader batch=1 collation
    print(f'benchmark sample: video={video_user_info[0]}, user={video_user_info[1]}, t={video_user_info[2]}')

    results = {}
    for mode in ('patch-selection', 'all-patch'):
        networking_head = NetworkingHead(input_dim=plm.hidden_size, output_dim=3, fut_window=20).to(DEVICE)
        pipeline = build_pipeline(mode, plm, networking_head, vit_model, patch_selection_module)
        pipeline.eval()

        stats = measure_inference_latency(pipeline, history, future, video_user_position,
                                            deadline_s=1.0, warmup=3, iters=10)
        results[mode] = stats
        print(format_latency_report(mode, stats))

    gap_ms = (results['patch-selection']['mean_s'] - results['all-patch']['mean_s']) * 1000
    print(f'\npatch-selection vs all-patch gap: {gap_ms:+.1f}ms '
          f'(negative = patch-selection is faster)')


if __name__ == '__main__':
    main()
