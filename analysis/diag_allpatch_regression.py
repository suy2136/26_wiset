"""
Investigate all-patch's LR-boosted-run regression (test MAE 40.55 vs original
23.92, despite validation loss improving throughout training):

1. rotation-aware MAE on the VALID set (not just the normalized MSE "Valid loss"
   the training loop reported) for the LR-boosted checkpoint -- separates
   "valid/test distribution mismatch" from "loss metric doesn't track MAE".
2. per-sample MAE distribution on the test set -- uniformly bad vs a few
   outliers dragging the mean up.
3. embed_multimodal weight drift: original (no LR boost) checkpoint vs this
   LR-boosted checkpoint, quantifying how much further the higher LR pushed it.
"""
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import cfg
from dataset.load_dataset import create_dataset
from utils.plms_utils import load_plm
from utils.normalize import normalize_data, denormalize_data
from utils.metrics import compute_mae, compute_each_mae
from models.networking_head import NetworkingHead
from models.low_rank import peft_model
from models.pipeline import Pipeline

import run_plm

DEVICE = 'cuda'
EMBED_SIZE = 4096
FUT_WINDOW = 20
RANK = 32
CKPT_ROOT = 'data/ft_plms/llama_base_low_rank/freeze_plm_False'
LR5_PREFIX = 'his_10_fut_20_ss_15_epochs_2_bs_32_lr_0.0002_seed_1_rank_32_scheduled_sampling_False'
ORIG_PREFIX = 'his_10_fut_20_ss_15_epochs_4_bs_32_lr_0.0002_seed_1_rank_32_scheduled_sampling_False'


def ckpt_path(prefix, mode='all-patch'):
    return os.path.join(CKPT_ROOT, f'multimodal_{mode}', 'Jin2022', '5Hz', prefix, 'best_model')


def build_pipeline(checkpoint_path):
    plm, tok, _ = load_plm('llama', os.path.join(cfg.plms_dir, 'llama', 'base'), plm_size='base',
                            device_input_side=DEVICE, device_output_side=DEVICE, device_middle_side=None)
    plm = peft_model(plm, 'llama', RANK)
    plm.set_networking_head(NetworkingHead(input_dim=EMBED_SIZE, output_dim=3, fut_window=FUT_WINDOW).to(DEVICE))
    import torchvision
    vit_model = torchvision.models.vit_b_16(pretrained=True).to(DEVICE)
    pipeline = Pipeline(plm, fut_window=FUT_WINDOW, device=DEVICE, embed_size=EMBED_SIZE, frequency=5,
                         multimodal_mode='all-patch', dataset='Jin2022', vit_model=vit_model)
    fake_args = SimpleNamespace(rank=RANK, use_adalora=False)
    pipeline = run_plm.load_model(fake_args, pipeline, checkpoint_path)
    pipeline.eval()
    return pipeline


def eval_split(pipeline, split, per_sample=False):
    raw = create_dataset('Jin2022', include=[split])[0]
    loader = DataLoader(raw, batch_size=1, shuffle=False)
    all_pred, all_gt = [], []
    with torch.no_grad():
        for history, future, video_user_info in loader:
            history, future = history.to(DEVICE), future.to(DEVICE)
            history_n = normalize_data(history, 'Jin2022')
            pred, gt = pipeline.inference(history_n, future, video_user_info)
            pred_deg = denormalize_data(pred, 'Jin2022')
            all_pred.append(pred_deg.cpu().numpy()[0])
            all_gt.append(gt.cpu().numpy()[0])
    all_pred, all_gt = np.array(all_pred), np.array(all_gt)
    mae = compute_mae(all_pred, all_gt, rotation=True)
    if per_sample:
        per_sample_err = compute_each_mae(all_pred, all_gt, rotation=True)
        return mae, per_sample_err
    return mae, None


print('=== #1: rotation-aware MAE on VALID set (LR-boosted checkpoint) ===')
pipeline_lr5 = build_pipeline(ckpt_path(LR5_PREFIX))
valid_mae, _ = eval_split(pipeline_lr5, 'valid')
print(f'LR-boosted all-patch: rotation-aware MAE on valid set = {valid_mae:.4f} deg')
print(f'(training-reported "Valid loss" at this checkpoint was 0.1375 -- normalized Tanh-space MSE, not directly comparable in units)')
print(f'(test-set MAE for this same checkpoint, already measured: 40.5518 deg)')
print()

print('=== #2: per-sample MAE distribution on TEST set (LR-boosted checkpoint) ===')
test_mae, per_sample_err = eval_split(pipeline_lr5, 'test', per_sample=True)
print(f'test MAE (recomputed, should match 40.5518): {test_mae:.4f}')
pcts = [50, 75, 90, 95, 99, 100]
print('Percentiles (deg):', {p: round(float(np.percentile(per_sample_err, p)), 2) for p in pcts})
worst_idx = np.argsort(per_sample_err)[-10:][::-1]
print('Top 10 worst samples (per-sample MAE, deg):', [round(float(per_sample_err[i]), 2) for i in worst_idx])
frac_over_90 = (per_sample_err > 90).mean()
frac_over_180 = (per_sample_err > 180).mean()
print(f'Fraction of samples with per-sample MAE > 90 deg: {frac_over_90:.4f}')
print(f'Fraction of samples with per-sample MAE > 180 deg (max possible with rotation wraparound is 180): {frac_over_180:.4f}')
median = np.median(per_sample_err)
mean = per_sample_err.mean()
print(f'median={median:.2f} mean={mean:.2f} (mean >> median implies outlier-driven; close implies uniformly bad)')

del pipeline_lr5
torch.cuda.empty_cache()
print()

print('=== #3: embed_multimodal drift -- original (no LR boost) vs LR-boosted checkpoint ===')
orig_state = torch.load(os.path.join(ckpt_path(ORIG_PREFIX), 'modules_except_plm.bin'), map_location=DEVICE)
lr5_state = torch.load(os.path.join(ckpt_path(LR5_PREFIX), 'modules_except_plm.bin'), map_location=DEVICE)

def sub_state(state, prefix):
    return {k[len(prefix)+1:]: v for k, v in state.items() if k.startswith(prefix + '.')}

orig_embed_multimodal = sub_state(orig_state, '1')
lr5_embed_multimodal = sub_state(lr5_state, '1')

for key in orig_embed_multimodal:
    o = orig_embed_multimodal[key].float()
    n = lr5_embed_multimodal[key].float()
    diff = (n - o).abs()
    rel = diff.mean().item() / (o.abs().mean().item() + 1e-8)
    print(f'  {key}: mean_abs_diff(orig vs lr5x)={diff.mean().item():.6f} '
          f'orig_mean_abs={o.abs().mean().item():.6f} lr5_mean_abs={n.abs().mean().item():.6f} '
          f'relative_change={rel:.4f} ({rel*100:.1f}%)')
