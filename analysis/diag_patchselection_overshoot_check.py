"""
Check whether patch-selection's apparent improvement trend (54.72->52.72->49.69
across the three experiments) is real, or the same LR-overshoot mechanism found
in all-patch just showing up less because patch-selection has fewer multimodal
tokens (avg ~8 vs all-patch's fixed 16):

1. embed_multimodal weight drift: original (no LR boost, 4-epoch) vs this run's
   LR-boosted (5x, 2-epoch) patch-selection checkpoint -- compare against
   all-patch's already-measured 28.8-30.5%.
2. rotation-aware MAE on the VALID set for the LR-boosted patch-selection
   checkpoint, vs its test-set MAE (49.69, already measured) and its
   training-reported best "Valid loss" (0.1343 at step 4410, normalized
   Tanh-space MSE).
"""
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

import run_plm

DEVICE = 'cuda'
EMBED_SIZE = 4096
FUT_WINDOW = 20
RANK = 32
CKPT_ROOT = 'data/ft_plms/llama_base_low_rank/freeze_plm_False'
LR5_PREFIX = 'his_10_fut_20_ss_15_epochs_2_bs_32_lr_0.0002_seed_1_rank_32_scheduled_sampling_False'
ORIG_PREFIX = 'his_10_fut_20_ss_15_epochs_4_bs_32_lr_0.0002_seed_1_rank_32_scheduled_sampling_False'
PATCH_SELECTION_WEIGHTS = 'data/models/patch_selection/best_patch_selection.pth'


def ckpt_path(prefix, mode='patch-selection'):
    return os.path.join(CKPT_ROOT, f'multimodal_{mode}', 'Jin2022', '5Hz', prefix, 'best_model')


print('=== #1: embed_multimodal drift -- original vs LR-boosted patch-selection checkpoint ===')
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
    print(f'  {key}: mean_abs_diff={diff.mean().item():.6f} orig_mean_abs={o.abs().mean().item():.6f} '
          f'lr5_mean_abs={n.abs().mean().item():.6f} relative_change={rel:.4f} ({rel*100:.1f}%)')
print('  (for comparison, all-patch measured 28.8% / 30.5% for weight / bias)')
print()

print('=== #2: rotation-aware MAE on VALID set (LR-boosted patch-selection checkpoint) ===')
plm, tok, _ = load_plm('llama', os.path.join(cfg.plms_dir, 'llama', 'base'), plm_size='base',
                        device_input_side=DEVICE, device_output_side=DEVICE, device_middle_side=None)
plm = peft_model(plm, 'llama', RANK)
plm.set_networking_head(NetworkingHead(input_dim=EMBED_SIZE, output_dim=3, fut_window=FUT_WINDOW).to(DEVICE))
import torchvision
vit_model = torchvision.models.vit_b_16(pretrained=True).to(DEVICE)
patch_selection_module = PatchSelectionModule(grid_rows=cfg.default_patch_grid[0], grid_cols=cfg.default_patch_grid[1]).to(DEVICE)
patch_selection_module.load_state_dict(torch.load(PATCH_SELECTION_WEIGHTS, map_location=DEVICE))
patch_selection_module.eval()
pipeline = Pipeline(plm, fut_window=FUT_WINDOW, device=DEVICE, embed_size=EMBED_SIZE, frequency=5,
                     multimodal_mode='patch-selection', dataset='Jin2022', vit_model=vit_model,
                     patch_selection_module=patch_selection_module)
fake_args = SimpleNamespace(rank=RANK, use_adalora=False)
pipeline = run_plm.load_model(fake_args, pipeline, ckpt_path(LR5_PREFIX))
pipeline.eval()

raw_valid = create_dataset('Jin2022', include=['valid'])[0]
loader = DataLoader(raw_valid, batch_size=1, shuffle=False)
import numpy as np
all_pred, all_gt = [], []
with torch.no_grad():
    for history, future, video_user_info in loader:
        history, future = history.to(DEVICE), future.to(DEVICE)
        history_n = normalize_data(history, 'Jin2022')
        pred, gt = pipeline.inference(history_n, future, video_user_info)
        pred_deg = denormalize_data(pred, 'Jin2022')
        all_pred.append(pred_deg.cpu().numpy()[0])
        all_gt.append(gt.cpu().numpy()[0])
valid_mae = compute_mae(np.array(all_pred), np.array(all_gt), rotation=True)
print(f'LR-boosted patch-selection: rotation-aware MAE on valid set = {valid_mae:.4f} deg')
print(f'(training-reported best "Valid loss" was 0.13425917266427348 at step 4410)')
print(f'(test-set MAE for this same checkpoint, already measured: 49.6878 deg)')
