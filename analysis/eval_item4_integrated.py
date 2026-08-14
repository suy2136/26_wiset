"""
Item 4: best mode (baseline, MAE=20.76 deg, lowest of the 3-condition comparison
in analysis/eval_3condition.py) + Selector (RecentK) + Speculative decoding,
evaluated on the same full 1,698-sample Jin2022 test set, so "pure patch-
selection effect" (item 3) and "final effect with the team's full optimization
stack on top of the best mode" (this item) are directly comparable at the same
scale. Reports MAE, forward count, and latency for:
  A. direct Pipeline.inference()                              (= item 3's baseline row)
  B. LlamaSelectablePipeline(RecentKSelector(k=6))
  C. LlamaSpeculativeBlockVerifyPipeline(selector=None)
  D. LlamaSpeculativeBlockVerifyPipeline(RecentKSelector(k=6))  (combined, "final")
"""
import sys
import os
import time
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
from models.selectable_pipeline import LlamaSelectablePipeline
from models.selectors import RecentKSelector
from models.speculative_pipeline import LlamaSpeculativeBlockVerifyPipeline

import run_plm

DEVICE = 'cuda'
EMBED_SIZE = 4096
FUT_WINDOW = 20
RANK = 32
RECENT_K = 6
GAMMA = 4
SPEC_THRESHOLD = 0.3  # same not-yet-calibrated starting point as the earlier smoke test; see report
CKPT_ROOT = 'data/ft_plms/llama_base_low_rank/freeze_plm_False'
FILE_PREFIX = 'his_10_fut_20_ss_15_epochs_4_bs_32_lr_0.0002_seed_1_rank_32_scheduled_sampling_False'
BEST_MODE = 'baseline'


def checkpoint_path(mode):
    return os.path.join(CKPT_ROOT, f'multimodal_{mode}', 'Jin2022', '5Hz', FILE_PREFIX, 'best_model')


def build_pipeline(mode):
    plm, tokenizer, _ = load_plm(
        'llama', os.path.join(cfg.plms_dir, 'llama', 'base'), plm_size='base',
        device_input_side=DEVICE, device_output_side=DEVICE, device_middle_side=None,
    )
    plm = peft_model(plm, 'llama', RANK)
    networking_head = NetworkingHead(input_dim=EMBED_SIZE, output_dim=3, fut_window=FUT_WINDOW).to(DEVICE)
    plm.set_networking_head(networking_head)
    pipeline = Pipeline(
        plm, fut_window=FUT_WINDOW, device=DEVICE, embed_size=EMBED_SIZE, frequency=5,
        multimodal_mode=mode, dataset='Jin2022', vit_model=None,
    )
    fake_args = SimpleNamespace(rank=RANK, use_adalora=False)
    pipeline = run_plm.load_model(fake_args, pipeline, checkpoint_path(mode))
    pipeline.eval()
    return pipeline


def eval_config(name, run_fn, loader):
    all_pred, all_gt = [], []
    forward_counts = []
    latencies_ms = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for i, (history, future, video_user_info) in enumerate(loader):
            history, future = history.to(DEVICE), future.to(DEVICE)
            history_n = normalize_data(history, 'Jin2022')
            torch.cuda.synchronize()
            ts = time.perf_counter()
            pred, fc = run_fn(history_n, future, video_user_info)
            torch.cuda.synchronize()
            latencies_ms.append((time.perf_counter() - ts) * 1000.0)
            pred_deg = denormalize_data(pred, 'Jin2022')
            all_pred.append(pred_deg.cpu().numpy()[0])
            all_gt.append(future.cpu().numpy()[0])
            forward_counts.append(fc)
            if (i + 1) % 400 == 0:
                running_mae = compute_mae(np.array(all_pred), np.array(all_gt), rotation=True)
                print(f'    [{name}] [{i+1}/{len(loader.dataset)}] running MAE={running_mae:.4f}, '
                      f'elapsed={time.perf_counter()-t0:.1f}s')
    mae = compute_mae(np.array(all_pred), np.array(all_gt), rotation=True)
    mean_fc = sum(forward_counts) / len(forward_counts)
    mean_lat = sum(latencies_ms) / len(latencies_ms)
    print(f'  [{name}] MAE={mae:.4f} deg, avg_forward_count={mean_fc:.2f}, latency={mean_lat:.2f} ms/sample')
    return mae, mean_fc, mean_lat


def main():
    print(f"=== Item 4: mode={BEST_MODE!r} + Selector/Speculative, full 1698-sample test ===")
    pipeline = build_pipeline(BEST_MODE)
    raw_test = create_dataset('Jin2022', include=['test'])[0]
    loader = DataLoader(raw_test, batch_size=1, shuffle=False)

    def run_direct(h, f, v):
        pred, gt = pipeline.inference(h, f, v)
        return pred, FUT_WINDOW

    def run_selector(h, f, v):
        wrapped = LlamaSelectablePipeline(pipeline, selector=RecentKSelector(k=RECENT_K))
        pred, gt = wrapped.inference(h, f, v)
        return pred, FUT_WINDOW

    def run_speculative(h, f, v):
        spec = LlamaSpeculativeBlockVerifyPipeline(pipeline, selector=None, gamma=GAMMA, acceptance_threshold=SPEC_THRESHOLD)
        pred, gt = spec.inference(h, f, v)
        return pred, spec.target_forward_count

    def run_combined(h, f, v):
        spec = LlamaSpeculativeBlockVerifyPipeline(pipeline, selector=RecentKSelector(k=RECENT_K), gamma=GAMMA, acceptance_threshold=SPEC_THRESHOLD)
        pred, gt = spec.inference(h, f, v)
        return pred, spec.target_forward_count

    results = {}
    results['A. direct'] = eval_config('A. direct', run_direct, loader)
    results[f'B. Selector RecentK(k={RECENT_K})'] = eval_config(f'B. Selector RecentK(k={RECENT_K})', run_selector, loader)
    results[f'C. Speculative (gamma={GAMMA}, th={SPEC_THRESHOLD})'] = eval_config(f'C. Speculative', run_speculative, loader)
    results['D. Selector + Speculative combined'] = eval_config('D. combined', run_combined, loader)

    peak_mem = torch.cuda.max_memory_allocated(DEVICE) / 1024**3
    print(f'\npeak GPU memory: {peak_mem:.2f} GiB')
    print('\n=== Summary ===')
    for name, (mae, fc, lat) in results.items():
        print(f'  {name:40s} MAE={mae:.4f} deg  forward_count={fc:.2f}  latency={lat:.2f} ms/sample')


if __name__ == '__main__':
    main()
