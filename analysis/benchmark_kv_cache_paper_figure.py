"""
Paper-figure benchmark: patch-selection latency WITHOUT the KV cache (the
pre-optimization auto_regressive(), which re-feeds the whole growing sequence to the
plm every step) vs WITH the KV cache (the current auto_regressive() in
models/pipeline.py, past_key_values + use_cache=True, plus the crop_patches_at()
optimization already in place).

This is a paired A/B on the *same* loaded llama-7b + patch_selection_module + sample,
in the same process, so the two numbers differ only in the auto_regressive loop
strategy -- not in model-loading jitter, sample choice, or GPU state.

Same conditions as analysis/benchmark_patch_crop_optimization.py and
IMPLEMENTATION_NOTES.md section 3: RTX 5090, llama-7b fp16, batch=1, warmup=3,
iters=10 (repeated across N_TRIALS fresh Pipeline instances to report run-to-run
noise, not just within-run std).

Run inside the vp_netllm env:
    /venv/vp_netllm/bin/python analysis/benchmark_kv_cache_paper_figure.py
"""
import json
import os
import sys
import time
import types

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
N_TRIALS = 3  # independent measurement rounds, each with its own warmup+iters
WARMUP = 3
ITERS = 10
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kv_cache_paper_figure_results.json')


def auto_regressive_no_kv_cache(self, x, future, video_user_position) -> torch.Tensor:
    """
    Reconstruction of the PRE-optimization auto_regressive(): no past_key_values, no
    use_cache. Every step re-feeds the entire accumulated sequence (history/image
    tokens + all previously generated future tokens) to self.plm from scratch, exactly
    as models/old/pipeline.py's EmbeddingModelViewportPrediction.forward() did. Kept
    here (rather than importing models/old) because models/old/pipeline.py's class
    predates the 3-way multimodal_mode dispatch and networking_head rename, so it
    can't be swapped in directly against the current Pipeline/get_multimodal_information.
    """
    history_viewports = x
    seq_len = x.shape[1]
    batch_embeddings = []
    for i in range(seq_len):
        batch_embeddings.append(self.embed_vp(self.conv1d(x[:, i, :]).view(1, 256)).unsqueeze(1))
    x = torch.cat(batch_embeddings, dim=1)

    if self.using_multimodal:
        mapped_tensor = self.get_multimodal_information(video_user_position, history_viewports)
        x = torch.cat([mapped_tensor, x], dim=1)

    x = self.embed_ln(x)

    outputlist = []
    plm_dtype = next(self.plm.parameters()).dtype
    for _ in range(self.fut_window_length):
        outputs = self.plm(inputs_embeds=x.to(plm_dtype),
                            attention_mask=torch.ones(x.shape[0], x.shape[1], dtype=torch.long, device=self.device))
        logits = outputs.logits.float()
        outputlist.append(logits)
        x = torch.cat((x, self.embed_vp(self.conv1d(logits)).unsqueeze(1)), dim=1)  # full sequence grows every step

    return torch.cat(outputlist, dim=1)


def build_pipeline(plm, networking_head, vit_model, patch_selection_module):
    plm.set_networking_head(networking_head)
    return Pipeline(
        plm, fut_window=20, device=DEVICE, embed_size=4096, frequency=cfg.default_dataset_frequency,
        multimodal_mode='patch-selection', dataset='Jin2022',
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
    video_user_position = [torch.tensor([v]) for v in video_user_info]
    print(f'benchmark sample: video={video_user_info[0]}, user={video_user_info[1]}, t={video_user_info[2]}')

    networking_head = NetworkingHead(input_dim=plm.hidden_size, output_dim=3, fut_window=20).to(DEVICE)
    pipeline = build_pipeline(plm, networking_head, vit_model, patch_selection_module)
    pipeline.eval()

    # sanity check: no-cache and with-cache paths must agree numerically before we trust
    # the latency comparison (same check IMPLEMENTATION_NOTES.md already ran once; redone
    # here against the *current* code, since this script binds fresh bound methods)
    with torch.no_grad():
        pred_cache, _ = pipeline.inference(history, future, video_user_position)
        pred_nocache = auto_regressive_no_kv_cache(pipeline, history, future, video_user_position)
    max_abs_diff = (pred_cache - pred_nocache).abs().max().item()
    print(f'numerical sanity check: max_abs_diff(with-cache, no-cache) = {max_abs_diff:.6f}')
    assert max_abs_diff < 1e-2, 'KV-cache and no-cache outputs diverge beyond fp16 tolerance -- do not trust latency numbers'

    results = {'no_kv_cache': [], 'with_kv_cache': []}
    for trial in range(N_TRIALS):
        # no-cache: temporarily rebind auto_regressive on this instance
        pipeline.auto_regressive = types.MethodType(auto_regressive_no_kv_cache, pipeline)
        stats_no_cache = measure_inference_latency(pipeline, history, future, video_user_position,
                                                     deadline_s=1.0, warmup=WARMUP, iters=ITERS)
        results['no_kv_cache'].append(stats_no_cache)
        print(f'[trial {trial}] ' + format_latency_report('no_kv_cache', stats_no_cache))

        # with-cache: restore the real (class-level) auto_regressive
        del pipeline.auto_regressive  # drop instance override, falls back to Pipeline.auto_regressive
        stats_with_cache = measure_inference_latency(pipeline, history, future, video_user_position,
                                                       deadline_s=1.0, warmup=WARMUP, iters=ITERS)
        results['with_kv_cache'].append(stats_with_cache)
        print(f'[trial {trial}] ' + format_latency_report('with_kv_cache', stats_with_cache))

    no_cache_means = [r['mean_s'] for r in results['no_kv_cache']]
    with_cache_means = [r['mean_s'] for r in results['with_kv_cache']]
    no_cache_avg = sum(no_cache_means) / len(no_cache_means) * 1000
    with_cache_avg = sum(with_cache_means) / len(with_cache_means) * 1000
    improvement_pct = (no_cache_avg - with_cache_avg) / no_cache_avg * 100

    summary = {
        'gpu': torch.cuda.get_device_name(0),
        'model': 'llama-7b fp16',
        'mode': 'patch-selection',
        'warmup': WARMUP, 'iters': ITERS, 'n_trials': N_TRIALS,
        'sanity_check_max_abs_diff': max_abs_diff,
        'no_kv_cache_ms_per_trial': [m * 1000 for m in no_cache_means],
        'with_kv_cache_ms_per_trial': [m * 1000 for m in with_cache_means],
        'no_kv_cache_ms_avg': no_cache_avg,
        'with_kv_cache_ms_avg': with_cache_avg,
        'absolute_saving_ms': no_cache_avg - with_cache_avg,
        'improvement_pct': improvement_pct,
        'no_kv_cache_peak_mem_mb': results['no_kv_cache'][-1]['peak_memory_mb'],
        'with_kv_cache_peak_mem_mb': results['with_kv_cache'][-1]['peak_memory_mb'],
    }
    with open(OUT_JSON, 'w') as f:
        json.dump(summary, f, indent=2)

    print('\n=== SUMMARY ===')
    print(f'no KV cache:   {no_cache_avg:.1f}ms avg over {N_TRIALS} trials {[f"{m:.1f}" for m in [x*1000 for x in no_cache_means]]}')
    print(f'with KV cache: {with_cache_avg:.1f}ms avg over {N_TRIALS} trials {[f"{m:.1f}" for m in [x*1000 for x in with_cache_means]]}')
    print(f'absolute saving: {no_cache_avg - with_cache_avg:.1f}ms, improvement: {improvement_pct:.1f}%')
    print(f'wrote {OUT_JSON}')


if __name__ == '__main__':
    main()
