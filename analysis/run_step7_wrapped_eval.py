"""
Step 7.3/7.4: load each mini-AdaLoRA-trained checkpoint (from step 7.2) and run
inference wrapped with LlamaSelectablePipeline / LlamaSpeculativeBlockVerifyPipeline,
on real held-out Jin2022 test samples, fp32 (matching training -- no dtype tricks
needed here since we never loaded a foreign fp16 checkpoint in this step).

For each multimodal_mode in {baseline, all-patch, patch-selection}, reports MAE
(denormalized, degrees), forward count, and latency for:
  A. direct Pipeline.inference()                        (no wrapper)
  B. LlamaSelectablePipeline(RecentKSelector(k=6))       (Selector only)
  C. LlamaSpeculativeBlockVerifyPipeline(selector=None)  (speculative only)
  D. LlamaSpeculativeBlockVerifyPipeline(RecentKSelector(k=6))  (combined)

Scope: smoke-test scale (NUM_EVAL_SAMPLES real Jin2022 test samples, the
1-epoch/24-sample AdaLoRA checkpoints from step 7.2 -- not real accuracy
numbers). The purpose is "does it run without crashing, is GPU memory sane,
is MAE in a plausible range (not NaN / not wildly worse than the unwrapped
baseline)" -- not a claim about which configuration is actually better.
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from types import SimpleNamespace

from config import cfg
from dataset.load_dataset import create_dataset
from utils.plms_utils import load_plm
from utils.normalize import normalize_data, denormalize_data
from models.networking_head import NetworkingHead
from models.low_rank import peft_model
from models.pipeline import Pipeline
from models.selectable_pipeline import LlamaSelectablePipeline
from models.selectors import RecentKSelector
from models.speculative_pipeline import LlamaSpeculativeBlockVerifyPipeline

import run_plm  # reuse its load_model() (handles the AdaLoRA checkpoint rank-resize)

DEVICE = 'cuda'
EMBED_SIZE = 4096
FUT_WINDOW = 20
RANK = 32
NUM_EVAL_SAMPLES = 10
RECENT_K = 6
GAMMA = 4
SPEC_THRESHOLD = 0.3  # not empirically calibrated for this checkpoint (see module docstring); a plausible starting point in the Tanh-bounded [-1,1] normalized output space

CKPT_ROOT = 'data/ft_plms/llama_base_low_rank_adalora/freeze_plm_False'
FILE_PREFIX = 'his_10_fut_20_ss_15_epochs_1_bs_1_lr_0.0002_seed_1_rank_32_scheduled_sampling_False'


def checkpoint_path(mode):
    return os.path.join(CKPT_ROOT, f'multimodal_{mode}', 'Jin2022', '5Hz', FILE_PREFIX, 'best_model')


def build_and_load_pipeline(mode):
    plm, tokenizer, _ = load_plm(
        'llama', os.path.join(cfg.plms_dir, 'llama', 'base'), plm_size='base',
        device_input_side=DEVICE, device_output_side=DEVICE, device_middle_side=None,
    )  # fp32, matches step 7.2 training (no --fp16)
    plm = peft_model(plm, 'llama', RANK, use_adalora=True, total_step=1)  # total_step unused at eval (no update_and_allocate call)
    networking_head = NetworkingHead(input_dim=EMBED_SIZE, output_dim=3, fut_window=FUT_WINDOW).to(DEVICE)
    plm.set_networking_head(networking_head)

    vit_model = None
    if mode in ('all-patch', 'patch-selection'):
        import torchvision
        vit_model = torchvision.models.vit_b_16(pretrained=True).to(DEVICE)

    pipeline = Pipeline(
        plm, fut_window=FUT_WINDOW, device=DEVICE, embed_size=EMBED_SIZE, frequency=5,
        multimodal_mode=mode, dataset='Jin2022', vit_model=vit_model,
    )

    fake_args = SimpleNamespace(rank=RANK, use_adalora=True)
    pipeline = run_plm.load_model(fake_args, pipeline, checkpoint_path(mode))
    pipeline.eval()
    return pipeline


def get_eval_samples(n=NUM_EVAL_SAMPLES):
    raw_test = create_dataset('Jin2022', include=['test'])[0]
    loader = DataLoader(raw_test, batch_size=1, shuffle=False)
    samples = []
    for i, (history, future, video_user_info) in enumerate(loader):
        if i >= n:
            break
        samples.append((history.to(DEVICE), future.to(DEVICE), video_user_info))
    return samples


def mae_degrees(pred, gt):
    pred_deg = denormalize_data(pred, 'Jin2022')
    return (pred_deg - gt).abs().mean().item()


def eval_config(name, run_fn, samples):
    maes = []
    forward_counts = []
    latencies_ms = []
    for history, future, video_user_info in samples:
        history_n = normalize_data(history, 'Jin2022')
        t0 = time.perf_counter()
        pred, fc = run_fn(history_n, future, video_user_info)
        torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        m = mae_degrees(pred, future)
        maes.append(m)
        forward_counts.append(fc)
    mae_mean = sum(maes) / len(maes)
    any_bad = any((not torch.isfinite(torch.tensor(m))) for m in maes)
    print(f"  [{name}] MAE={mae_mean:.3f} deg (n={len(samples)}, finite={not any_bad}), "
          f"avg_forward_count={sum(forward_counts)/len(forward_counts):.2f}, "
          f"latency={sum(latencies_ms)/len(latencies_ms):.1f} ms/sample")
    return mae_mean, any_bad


def run_mode(mode):
    print(f"=== multimodal_mode={mode!r} (real 7B, fp32, GPU, mini-AdaLoRA checkpoint) ===")
    pipeline = build_and_load_pipeline(mode)
    samples = get_eval_samples()

    with torch.no_grad():
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

        mae_direct, bad_direct = eval_config('A. direct (no wrapper)', run_direct, samples)
        mae_selector, bad_selector = eval_config(f'B. Selector RecentK(k={RECENT_K})', run_selector, samples)
        mae_spec, bad_spec = eval_config(f'C. Speculative (gamma={GAMMA}, th={SPEC_THRESHOLD})', run_speculative, samples)
        mae_combined, bad_combined = eval_config('D. Selector + Speculative combined', run_combined, samples)

    peak_mem = torch.cuda.max_memory_allocated(DEVICE) / 1024**3
    print(f"  peak GPU memory this mode: {peak_mem:.2f} GiB")

    for name, mae, bad in (('B', mae_selector, bad_selector), ('C', mae_spec, bad_spec), ('D', mae_combined, bad_combined)):
        if bad or mae > mae_direct * 3:
            print(f"  \033[31m[FLAG]\033[0m config {name} MAE={mae:.3f} vs direct={mae_direct:.3f} "
                  f"-- more than 3x worse or non-finite, stopping scope per instructions")
    print()
    del pipeline
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(DEVICE)


if __name__ == "__main__":
    for mode in ("baseline", "all-patch", "patch-selection"):
        run_mode(mode)
    print("Step 7.3/7.4 wrapped-inference smoke eval done.")
