"""
Item 1 of the 3-condition comparison request: reproduce Soyun's non-multimodal
baseline (her PHASE_B_REAL_RESULTS.md row "A. baseline") on the SAME
try_llama2_7b checkpoint, full 1,698-sample Jin2022 test set, sample by sample,
using OUR OWN harness (models/pipeline.py::Pipeline, not her scripts).

try_llama2_7b was trained non-multimodal (using_multimodal=False throughout
Soyun_ModuleHead's docs) -- this is our project's "text-only" configuration
(multimodal_mode='none'), NOT our own multimodal_mode='baseline' (single-CLS-
token) arm of the 3-way comparison. Using multimodal_mode='baseline' here would
feed an untrained embed_multimodal into a checkpoint that never saw it, and
would fail to reproduce Soyun's number for the wrong reason.

Reference numbers (Soyun's PHASE_B_REAL_RESULTS.md):
  Full 1,698-sample baseline MAE: 12.798559 (this session's harness)
  vs. the 7.26 report's 12.798525 (diff 0.000034, fp16 noise)
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
from models.networking_head import NetworkingHead
from models.low_rank import peft_model
from models.pipeline import Pipeline
from utils.metrics import compute_mae

DEVICE = 'cuda'
EMBED_SIZE = 4096
FUT_WINDOW = 20
RANK = 32
CHECKPOINT_PATH = '/workspace/data/ft_plms/try_llama2_7b'
REFERENCE_MAE_THIS_SESSION = 12.798559
REFERENCE_MAE_726 = 12.798525


def build_pipeline():
    plm, tokenizer, _ = load_plm(
        'llama', os.path.join(cfg.plms_dir, 'llama', 'base'), plm_size='base',
        device_input_side=DEVICE, device_output_side=DEVICE, device_middle_side=None,
        torch_dtype=torch.float16,
    )
    plm = peft_model(plm, 'llama', RANK)

    # Same RMSNorm dtype fix as analysis/verify_real_7b_equivalence.py -- see that
    # file's comments for the full root-cause (peft_model() upcasts 1-D params to
    # fp32, which promotes fp16 hidden_states to fp32 inside LlamaRMSNorm and
    # crashes the next fp16 Linear).
    for name, param in plm.named_parameters():
        if 'norm.weight' in name:
            param.data = param.data.to(torch.float16)

    networking_head = NetworkingHead(input_dim=EMBED_SIZE, output_dim=3, fut_window=FUT_WINDOW).to(DEVICE)
    plm.set_networking_head(networking_head)

    pipeline = Pipeline(
        plm, fut_window=FUT_WINDOW, device=DEVICE, embed_size=EMBED_SIZE, frequency=5,
        multimodal_mode='none', dataset='Jin2022', vit_model=None,
    )

    pipeline.plm.load_adapter(CHECKPOINT_PATH, adapter_name='default')
    pipeline.plm.set_adapter('default')
    state = torch.load(os.path.join(CHECKPOINT_PATH, 'modules_except_plm.bin'), map_location=DEVICE)
    state = {k.replace('task_head', 'networking_head'): v for k, v in state.items()}
    incompatible = pipeline.modules_except_plm.load_state_dict(state, strict=True)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys, incompatible
    pipeline.eval()
    return pipeline


def main():
    pipeline = build_pipeline()
    raw_test = create_dataset('Jin2022', include=['test'])[0]
    loader = DataLoader(raw_test, batch_size=1, shuffle=False)
    print(f'Full test set size: {len(raw_test)}')

    # compute_mae(..., rotation=True) matches utils/result_notebook.py's own
    # aggregation exactly (circular/wraparound-aware angular distance -- a naive
    # .abs().mean() overcounts samples whose yaw crosses the 0/360 boundary, e.g.
    # pred=359 vs gt=1 should be a 2-degree error, not 358). Accumulate raw
    # (pred, gt) arrays and call compute_mae once at the end, same as
    # ResultNotebook.write()'s "-1,-1" aggregate-over-all-pairs row, rather than
    # averaging per-sample MAEs (mean-of-means != true aggregate mean here,
    # though the difference is negligible since np.mean over all samples * fut_window*3
    # elements is what both approaches converge to -- kept exact to avoid any doubt).
    all_pred, all_gt = [], []
    t0 = time.perf_counter()
    with torch.no_grad():
        for i, (history, future, video_user_info) in enumerate(loader):
            history, future = history.to(DEVICE), future.to(DEVICE)
            history_n = normalize_data(history, 'Jin2022')
            pred, gt = pipeline.inference(history_n, future, video_user_info)
            pred_deg = denormalize_data(pred, 'Jin2022')
            all_pred.append(pred_deg.cpu().numpy()[0])
            all_gt.append(gt.cpu().numpy()[0])
            if (i + 1) % 200 == 0:
                running_mae = compute_mae(np.array(all_pred), np.array(all_gt), rotation=True)
                elapsed = time.perf_counter() - t0
                print(f'  [{i+1}/{len(raw_test)}] running MAE={running_mae:.6f} deg, elapsed={elapsed:.1f}s')

    mae = compute_mae(np.array(all_pred), np.array(all_gt), rotation=True)
    elapsed = time.perf_counter() - t0
    print(f'\nFinal MAE over {len(all_pred)} samples: {mae:.6f} deg (elapsed {elapsed:.1f}s)')
    print(f'Reference (this-session harness, Soyun PHASE_B_REAL_RESULTS.md): {REFERENCE_MAE_THIS_SESSION}')
    print(f'Reference (7.26 report): {REFERENCE_MAE_726}')
    print(f'Diff vs this-session reference: {abs(mae - REFERENCE_MAE_THIS_SESSION):.6f}')
    print(f'Diff vs 7.26 reference: {abs(mae - REFERENCE_MAE_726):.6f}')


if __name__ == '__main__':
    main()
