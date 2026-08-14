"""
[Task 3] Patch-selection heatmap analysis.

Runs the current best_patch_selection.pth checkpoint through select_patches()
on the full Jin2022 held-out TEST split (the official split from config.py,
never used for training/validation), using the exact default inference
settings (no --patch-top-k / --patch-threshold override -> threshold=0.5,
plus the pipeline's "always keep >=1 patch" fallback), and records which of
the 16 grid patches get selected. Produces a selection-frequency heatmap and
a text summary.

Run inside the vp_netllm conda env:
    conda activate vp_netllm && python analysis/patch_selection_heatmap.py
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import cfg
from dataset.load_dataset import create_dataset
from models.patch_selection import PatchSelectionModule

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = '/workspace/data/models/patch_selection/best_patch_selection.pth'
GRID_ROWS, GRID_COLS = cfg.default_patch_grid
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Mirrors run_plm.py CLI defaults: neither --patch-top-k nor --patch-threshold
# passed -> select_patches() falls back to threshold=0.5.
PATCH_TOP_K = None
PATCH_THRESHOLD = None


def main():
    print(f'device: {DEVICE}')

    (test_ds,) = create_dataset('Jin2022', include=('test',))
    print(f'held-out TEST split: {len(test_ds)} samples '
          f'(videos={cfg.dataset_video_split["Jin2022"]["test"]}, '
          f'users={len(cfg.dataset_user_split["Jin2022"]["test"])})')

    module = PatchSelectionModule(grid_rows=GRID_ROWS, grid_cols=GRID_COLS).to(DEVICE)
    state_dict = torch.load(CKPT_PATH, map_location=DEVICE)
    module.load_state_dict(state_dict)
    module.eval()
    print(f'loaded checkpoint: {CKPT_PATH}')

    histories = np.stack([test_ds[i][0] for i in range(len(test_ds))])  # (N, T, 3)
    metas = [test_ds[i][2] for i in range(len(test_ds))]  # (video, user, timestep)
    hist_t = torch.tensor(histories, dtype=torch.float32, device=DEVICE)

    num_patches = GRID_ROWS * GRID_COLS
    selection_counts = np.zeros(num_patches, dtype=np.int64)
    num_selected_per_sample = np.zeros(len(test_ds), dtype=np.int64)
    all_selected = []

    batch_size = 256
    with torch.no_grad():
        for start in range(0, len(test_ds), batch_size):
            end = min(start + batch_size, len(test_ds))
            logits = module(hist_t[start:end])  # (b, num_patches)
            mask = module.select_patches(logits, top_k=PATCH_TOP_K, threshold=PATCH_THRESHOLD)  # (b, num_patches)
            for i in range(end - start):
                idx = mask[i].nonzero(as_tuple=True)[0]
                if idx.numel() == 0:  # pipeline.py's "always feed >=1 patch" fallback
                    idx = logits[i].argmax().unsqueeze(0)
                idx = idx.cpu().numpy()
                selection_counts[idx] += 1
                num_selected_per_sample[start + i] = len(idx)
                all_selected.append(idx.tolist())

    n = len(test_ds)
    selection_pct = selection_counts / n * 100.0

    # ---- frequency table ----
    rows = []
    for p in range(num_patches):
        r, c = divmod(p, GRID_COLS)
        rows.append({
            'patch_index': p, 'row': r, 'col': c,
            'count': int(selection_counts[p]), 'pct': round(float(selection_pct[p]), 2),
        })
    rows_sorted = sorted(rows, key=lambda x: -x['pct'])

    print('\n=== selection frequency (sorted desc) ===')
    print(f'{"patch":>5} {"(row,col)":>10} {"count":>7} {"pct":>7}')
    for r in rows_sorted:
        rc = f'({r["row"]},{r["col"]})'
        print(f'{r["patch_index"]:>5} {rc:>10} {r["count"]:>7} {r["pct"]:>6.2f}%')

    print(f'\nnum_selected per sample: min={num_selected_per_sample.min()}, '
          f'max={num_selected_per_sample.max()}, mean={num_selected_per_sample.mean():.2f}, '
          f'median={int(np.median(num_selected_per_sample))}')
    print('num_selected distribution:', dict(zip(*np.unique(num_selected_per_sample, return_counts=True))))

    summary = {
        'checkpoint': CKPT_PATH,
        'split': 'test',
        'dataset': 'Jin2022',
        'num_samples': n,
        'grid_rows': GRID_ROWS, 'grid_cols': GRID_COLS,
        'patch_top_k': PATCH_TOP_K, 'patch_threshold': PATCH_THRESHOLD,
        'effective_threshold': 0.5,
        'frequency_table': rows_sorted,
        'num_selected_stats': {
            'min': int(num_selected_per_sample.min()),
            'max': int(num_selected_per_sample.max()),
            'mean': float(num_selected_per_sample.mean()),
            'median': int(np.median(num_selected_per_sample)),
        },
    }
    out_json = os.path.join(OUT_DIR, 'patch_selection_frequency.json')
    with open(out_json, 'w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f'\nsaved frequency table -> {out_json}')

    np.save(os.path.join(OUT_DIR, 'patch_selection_freq_grid.npy'),
            selection_pct.reshape(GRID_ROWS, GRID_COLS))
    print(f'saved freq grid (npy) -> {os.path.join(OUT_DIR, "patch_selection_freq_grid.npy")}')


if __name__ == '__main__':
    main()
