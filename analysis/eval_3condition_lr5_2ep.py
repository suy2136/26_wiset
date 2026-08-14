"""
Item 3: evaluate the 3-condition comparison (baseline/all-patch/patch-selection,
standard LoRA rank=32, 4 epochs -- see analysis/eval_baseline_repro.py's sibling
training runs) on the SAME full 1,698-sample Jin2022 test set Soyun used, so all
three are compared at the same scale. Uses the project's own rotation-aware
compute_mae(..., rotation=True) throughout (see eval_baseline_repro.py's fix for
why a naive .abs().mean() overcounts errors near the 0/360 degree boundary).

Loads each mode's own best_model checkpoint (selected by validation loss during
training -- for baseline that's the epoch-1 checkpoint, for all-patch epoch-3,
for patch-selection epoch-1; see the training reports for why).
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import cfg
from dataset.load_dataset import create_dataset
from utils.plms_utils import load_plm
from utils.normalize import normalize_data, denormalize_data
from utils.metrics import compute_mae
from models.networking_head import NetworkingHead
from models.low_rank import peft_model
from models.pipeline import Pipeline
from models.patch_selection import PatchSelectionModule

import run_plm  # reuse its load_model()

DEVICE = 'cuda'
EMBED_SIZE = 4096
FUT_WINDOW = 20
RANK = 32
CKPT_ROOT = 'data/ft_plms/llama_base_low_rank/freeze_plm_False'
FILE_PREFIX = 'his_10_fut_20_ss_15_epochs_2_bs_32_lr_0.0002_seed_1_rank_32_scheduled_sampling_False'
PATCH_SELECTION_WEIGHTS = 'data/models/patch_selection/best_patch_selection.pth'


def checkpoint_path(mode):
    return os.path.join(CKPT_ROOT, f'multimodal_{mode}', 'Jin2022', '5Hz', FILE_PREFIX, 'best_model')


def build_pipeline(mode):
    from types import SimpleNamespace
    plm, tokenizer, _ = load_plm(
        'llama', os.path.join(cfg.plms_dir, 'llama', 'base'), plm_size='base',
        device_input_side=DEVICE, device_output_side=DEVICE, device_middle_side=None,
    )  # fp32, matches how these checkpoints were trained (no --fp16)
    plm = peft_model(plm, 'llama', RANK)  # plain LoRA, matches training (use_adalora=False)
    networking_head = NetworkingHead(input_dim=EMBED_SIZE, output_dim=3, fut_window=FUT_WINDOW).to(DEVICE)
    plm.set_networking_head(networking_head)

    vit_model = None
    if mode in ('all-patch', 'patch-selection'):
        import torchvision
        vit_model = torchvision.models.vit_b_16(pretrained=True).to(DEVICE)

    patch_selection_module = None
    if mode == 'patch-selection':
        patch_selection_module = PatchSelectionModule(
            grid_rows=cfg.default_patch_grid[0], grid_cols=cfg.default_patch_grid[1]).to(DEVICE)
        patch_selection_module.load_state_dict(torch.load(PATCH_SELECTION_WEIGHTS, map_location=DEVICE))
        patch_selection_module.eval()

    pipeline = Pipeline(
        plm, fut_window=FUT_WINDOW, device=DEVICE, embed_size=EMBED_SIZE, frequency=5,
        multimodal_mode=mode, dataset='Jin2022', vit_model=vit_model,
        patch_selection_module=patch_selection_module,
    )

    fake_args = SimpleNamespace(rank=RANK, use_adalora=False)
    pipeline = run_plm.load_model(fake_args, pipeline, checkpoint_path(mode))
    pipeline.eval()
    return pipeline


def run_mode(mode):
    print(f"=== multimodal_mode={mode!r} (real 7B, fp32, GPU, 4-epoch checkpoint, full 1698-sample test) ===")
    pipeline = build_pipeline(mode)
    raw_test = create_dataset('Jin2022', include=['test'])[0]
    loader = DataLoader(raw_test, batch_size=1, shuffle=False)

    all_pred, all_gt = [], []
    latencies_ms = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for i, (history, future, video_user_info) in enumerate(loader):
            history, future = history.to(DEVICE), future.to(DEVICE)
            history_n = normalize_data(history, 'Jin2022')
            torch.cuda.synchronize()
            ts = time.perf_counter()
            pred, gt = pipeline.inference(history_n, future, video_user_info)
            torch.cuda.synchronize()
            latencies_ms.append((time.perf_counter() - ts) * 1000.0)
            pred_deg = denormalize_data(pred, 'Jin2022')
            all_pred.append(pred_deg.cpu().numpy()[0])
            all_gt.append(gt.cpu().numpy()[0])
            if (i + 1) % 200 == 0:
                running_mae = compute_mae(np.array(all_pred), np.array(all_gt), rotation=True)
                elapsed = time.perf_counter() - t0
                print(f'  [{i+1}/{len(raw_test)}] running MAE={running_mae:.4f} deg, elapsed={elapsed:.1f}s')

    mae = compute_mae(np.array(all_pred), np.array(all_gt), rotation=True)
    mean_latency = sum(latencies_ms) / len(latencies_ms)
    peak_mem = torch.cuda.max_memory_allocated(DEVICE) / 1024**3
    print(f'\n[{mode}] Final MAE over {len(all_pred)} samples: {mae:.6f} deg, '
          f'mean latency={mean_latency:.2f} ms/sample, peak_mem={peak_mem:.2f} GiB')
    del pipeline
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(DEVICE)
    print()
    return mae, mean_latency


if __name__ == '__main__':
    results = {}
    for mode in ('baseline', 'all-patch', 'patch-selection'):
        results[mode] = run_mode(mode)
    print('=== Summary ===')
    for mode, (mae, lat) in results.items():
        print(f'  {mode:16s} MAE={mae:.4f} deg  latency={lat:.2f} ms/sample')
