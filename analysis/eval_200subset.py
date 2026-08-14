"""
Quick 200-sample test-subset MAE check (same first-200 pattern used in the
patch-selection dropout/drift diagnostics) for the differential-LR direction
check. Not a final-number eval -- just enough to see whether patch-selection
moves toward all-patch after --multimodal-lr-multiplier.
"""
import sys
import os
import argparse
from types import SimpleNamespace

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

import run_plm

DEVICE = 'cuda'
EMBED_SIZE = 4096
FUT_WINDOW = 20
RANK = 32
N_SAMPLES = 200
PATCH_SELECTION_WEIGHTS = 'data/models/patch_selection/best_patch_selection.pth'


def build_pipeline(mode, checkpoint_path):
    plm, tokenizer, _ = load_plm(
        'llama', os.path.join(cfg.plms_dir, 'llama', 'base'), plm_size='base',
        device_input_side=DEVICE, device_output_side=DEVICE, device_middle_side=None,
    )
    plm = peft_model(plm, 'llama', RANK)
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
    pipeline = run_plm.load_model(fake_args, pipeline, checkpoint_path)
    pipeline.eval()
    return pipeline


def main(mode, checkpoint_path):
    pipeline = build_pipeline(mode, checkpoint_path)
    raw_test = create_dataset('Jin2022', include=['test'])[0]
    loader = DataLoader(raw_test, batch_size=1, shuffle=False)

    all_pred, all_gt = [], []
    with torch.no_grad():
        for i, (history, future, video_user_info) in enumerate(loader):
            if i >= N_SAMPLES:
                break
            history, future = history.to(DEVICE), future.to(DEVICE)
            history_n = normalize_data(history, 'Jin2022')
            pred, gt = pipeline.inference(history_n, future, video_user_info)
            pred_deg = denormalize_data(pred, 'Jin2022')
            all_pred.append(pred_deg.cpu().numpy()[0])
            all_gt.append(gt.cpu().numpy()[0])

    mae = compute_mae(np.array(all_pred), np.array(all_gt), rotation=True)
    print(f'RESULT mode={mode} n={len(all_pred)} MAE={mae:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True)
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    main(args.mode, args.checkpoint)
