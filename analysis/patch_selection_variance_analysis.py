"""
[Task 4] Does PatchSelectionModule react to trajectory, or has it collapsed to a
fixed handful of patches?

Core question: within the SAME video, across different users/timesteps (i.e.
different historical trajectories), does the selected patch set change
meaningfully, or is it essentially constant?

Pipeline:
  1. Re-run the current best_patch_selection.pth over the full Jin2022 TEST
     split (same setup as task 3: threshold=0.5 default + >=1-patch fallback),
     recording per-sample selected patch sets.
  2. Per-video variance: pairwise Jaccard similarity of selected sets across
     all (user, timestep) samples in that video, split into same-user
     (different timestep) vs cross-user pairs; exact-match rate against the
     video's modal selection (collapse indicator); per-patch selection-rate
     range (fixed vs variable patches).
  3. Trajectory feature extraction per sample: current grid position, net
     movement direction, mean angular speed, stillness (data-driven tertile
     threshold on speed).
  4. Relate trajectory features to selection: current-position inclusion
     rate; "ahead" vs "behind" neighbor selection rate for moving vs still
     samples; num_selected by movement state.

Run inside the vp_netllm conda env:
    conda activate vp_netllm && python analysis/patch_selection_variance_analysis.py
"""
import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import cfg
from dataset.load_dataset import create_dataset
from models.patch_selection import PatchSelectionModule
from utils.patch_labeling import yaw_pitch_to_grid_cell, grid_cell_to_patch_index

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = '/workspace/data/models/patch_selection/best_patch_selection.pth'
GRID_ROWS, GRID_COLS = cfg.default_patch_grid
NUM_PATCHES = GRID_ROWS * GRID_COLS
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PATCH_TOP_K = None
PATCH_THRESHOLD = None


def wrap180(deg):
    """wrap an angle delta into [-180, 180]"""
    return (deg + 180.0) % 360.0 - 180.0


def run_inference():
    (test_ds,) = create_dataset('Jin2022', include=('test',))
    n = len(test_ds)
    histories = np.stack([test_ds[i][0] for i in range(n)])  # (N, T, 3) roll,pitch,yaw
    metas = [test_ds[i][2] for i in range(n)]  # (video, user, timestep)

    module = PatchSelectionModule(grid_rows=GRID_ROWS, grid_cols=GRID_COLS).to(DEVICE)
    module.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    module.eval()

    hist_t = torch.tensor(histories, dtype=torch.float32, device=DEVICE)
    mask_all = np.zeros((n, NUM_PATCHES), dtype=bool)
    logits_all = np.zeros((n, NUM_PATCHES), dtype=np.float32)
    batch_size = 256
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            logits = module(hist_t[start:end])
            mask = module.select_patches(logits, top_k=PATCH_TOP_K, threshold=PATCH_THRESHOLD)
            logits_np = logits.cpu().numpy()
            mask_np = mask.cpu().numpy()
            for i in range(end - start):
                if not mask_np[i].any():
                    mask_np[i, logits_np[i].argmax()] = True
            mask_all[start:end] = mask_np
            logits_all[start:end] = logits_np

    return histories, metas, mask_all, logits_all


def extract_trajectory_features(histories):
    """
    histories: (N, T, 3) columns = roll, pitch, yaw (degrees)
    returns dict of per-sample arrays.
    """
    N, T, _ = histories.shape
    pitch = histories[:, :, 1]
    yaw = histories[:, :, 2]

    dyaw_steps = wrap180(yaw[:, 1:] - yaw[:, :-1])   # (N, T-1)
    dpitch_steps = pitch[:, 1:] - pitch[:, :-1]        # (N, T-1)
    step_speed = np.sqrt(dyaw_steps ** 2 + dpitch_steps ** 2)  # (N, T-1)
    mean_speed = step_speed.mean(axis=1)  # deg/step

    # Net displacement = SUM of the per-step wrapped deltas (the cumulative rotation
    # actually traversed), NOT wrap180(last - first). The latter collapses to the
    # shortest angular path between endpoints and silently flips sign whenever the
    # true cumulative swing exceeds 180 deg (common for the fastest trajectories,
    # i.e. exactly the samples this analysis cares most about) - e.g. a steady
    # -209 deg continuous left turn would otherwise be reported as a +151 deg
    # "right" turn.
    dyaw_net = dyaw_steps.sum(axis=1)
    dpitch_net = pitch[:, -1] - pitch[:, 0]  # pitch never wraps, no correction needed

    last_yaw, last_pitch = yaw[:, -1], pitch[:, -1]
    cur_row, cur_col = yaw_pitch_to_grid_cell(last_yaw, last_pitch, GRID_ROWS, GRID_COLS)
    cur_patch = grid_cell_to_patch_index(cur_row, cur_col, GRID_COLS)

    return {
        'mean_speed': mean_speed, 'dyaw_net': dyaw_net, 'dpitch_net': dpitch_net,
        'cur_row': cur_row, 'cur_col': cur_col, 'cur_patch': cur_patch,
    }


def classify_direction(dyaw_net, dpitch_net, mean_speed, still_thresh):
    N = len(mean_speed)
    direction = np.empty(N, dtype=object)
    still = mean_speed < still_thresh
    direction[still] = 'still'
    moving = ~still
    horiz = np.abs(dyaw_net) >= np.abs(dpitch_net)
    direction[moving & horiz & (dyaw_net > 0)] = 'right'
    direction[moving & horiz & (dyaw_net <= 0)] = 'left'
    direction[moving & ~horiz & (dpitch_net > 0)] = 'up'
    direction[moving & ~horiz & (dpitch_net <= 0)] = 'down'
    return direction, still


def jaccard_matrix(bool_mat):
    """bool_mat: (n, P) -> (n, n) pairwise jaccard similarity"""
    m = bool_mat.astype(np.int32)
    inter = m @ m.T
    row_sum = m.sum(axis=1)
    union = row_sum[:, None] + row_sum[None, :] - inter
    with np.errstate(divide='ignore', invalid='ignore'):
        jac = np.where(union > 0, inter / np.maximum(union, 1), 1.0)
    return jac


def analyze_per_video(metas, mask_all):
    videos = sorted(set(m[0] for m in metas))
    users_arr = np.array([m[1] for m in metas])
    videos_arr = np.array([m[0] for m in metas])

    report = {}
    for v in videos:
        idx = np.where(videos_arr == v)[0]
        sub_mask = mask_all[idx]
        sub_users = users_arr[idx]
        n_v = len(idx)

        jac = jaccard_matrix(sub_mask)
        iu = np.triu_indices(n_v, k=1)
        pair_jac = jac[iu]
        same_user = sub_users[iu[0]] == sub_users[iu[1]]

        # modal selected set / exact-match (collapse) rate
        keys = [tuple(row.nonzero()[0].tolist()) for row in sub_mask]
        counts = defaultdict(int)
        for k in keys:
            counts[k] += 1
        modal_key, modal_count = max(counts.items(), key=lambda kv: kv[1])

        # per-patch selection rate within this video
        patch_rate = sub_mask.mean(axis=0) * 100
        n_fixed = int(((patch_rate < 5) | (patch_rate > 95)).sum())
        n_variable = NUM_PATCHES - n_fixed

        report[int(v)] = {
            'n_samples': int(n_v),
            'n_unique_selected_sets': len(counts),
            'mean_pairwise_jaccard': float(pair_jac.mean()),
            'median_pairwise_jaccard': float(np.median(pair_jac)),
            'mean_pairwise_jaccard_same_user': float(pair_jac[same_user].mean()) if same_user.any() else None,
            'mean_pairwise_jaccard_cross_user': float(pair_jac[~same_user].mean()) if (~same_user).any() else None,
            'modal_set': list(modal_key),
            'modal_set_share_pct': round(modal_count / n_v * 100, 2),
            'n_fixed_patches_(<5%_or_>95%)': n_fixed,
            'n_variable_patches_(5-95%)': n_variable,
            'patch_selection_rate_pct': np.round(patch_rate, 1).tolist(),
        }
    return report


def analyze_relationships(metas, mask_all, feats, direction, still):
    N = len(metas)
    cur_patch = feats['cur_patch']

    cur_included = mask_all[np.arange(N), cur_patch]
    cur_inclusion_rate = cur_included.mean() * 100

    num_selected = mask_all.sum(axis=1)
    still_num = num_selected[still]
    moving_num = num_selected[~still]

    # "ahead" vs "behind" neighbor relative to movement direction
    dir_to_delta = {'right': (0, 1), 'left': (0, -1), 'up': (-1, 0), 'down': (1, 0)}
    ahead_sel, behind_sel = [], []
    for i in range(N):
        d = direction[i]
        if d not in dir_to_delta:
            continue
        dr, dc = dir_to_delta[d]
        r, c = feats['cur_row'][i], feats['cur_col'][i]
        ar, ac = r + dr, c + dc
        br, bc = r - dr, c - dc
        if 0 <= ar < GRID_ROWS:
            ac_w = ac % GRID_COLS
            ahead_idx = grid_cell_to_patch_index(ar, ac_w, GRID_COLS)
            ahead_sel.append(mask_all[i, ahead_idx])
        if 0 <= br < GRID_ROWS:
            bc_w = bc % GRID_COLS
            behind_idx = grid_cell_to_patch_index(br, bc_w, GRID_COLS)
            behind_sel.append(mask_all[i, behind_idx])

    # same comparison restricted to STILL samples as a control (direction is meaningless
    # but we bucket by the "would-be" direction from tiny net drift, for a baseline)
    return {
        'current_position_inclusion_rate_pct': round(float(cur_inclusion_rate), 2),
        'num_selected_still_mean': round(float(still_num.mean()), 2) if still.any() else None,
        'num_selected_moving_mean': round(float(moving_num.mean()), 2) if (~still).any() else None,
        'num_selected_still_median': int(np.median(still_num)) if still.any() else None,
        'num_selected_moving_median': int(np.median(moving_num)) if (~still).any() else None,
        'ahead_neighbor_selection_rate_pct': round(float(np.mean(ahead_sel) * 100), 2) if ahead_sel else None,
        'behind_neighbor_selection_rate_pct': round(float(np.mean(behind_sel) * 100), 2) if behind_sel else None,
        'n_moving_samples_with_valid_ahead_behind': len(ahead_sel),
    }


def main():
    print(f'device: {DEVICE}')
    histories, metas, mask_all, logits_all = run_inference()
    n = len(metas)
    print(f'inference done on {n} test samples')

    feats = extract_trajectory_features(histories)

    speed_p33, speed_p67 = np.percentile(feats['mean_speed'], [33, 67])
    still_thresh = speed_p33
    print(f'\nspeed distribution (deg/step): p33={speed_p33:.3f}, p50={np.median(feats["mean_speed"]):.3f}, '
          f'p67={speed_p67:.3f}, max={feats["mean_speed"].max():.3f}')
    print(f'stillness threshold (data-driven, p33 of mean_speed): {still_thresh:.3f} deg/step')

    direction, still = classify_direction(feats['dyaw_net'], feats['dpitch_net'], feats['mean_speed'], still_thresh)
    dir_vals, dir_counts = np.unique(direction, return_counts=True)
    print('direction distribution:', dict(zip(dir_vals, dir_counts.tolist())))

    print('\n=== per-video variance / collapse analysis ===')
    per_video = analyze_per_video(metas, mask_all)
    for v, r in per_video.items():
        print(f'\nvideo {v}: n={r["n_samples"]}, unique_selected_sets={r["n_unique_selected_sets"]}')
        print(f'  mean pairwise Jaccard: {r["mean_pairwise_jaccard"]:.3f} '
              f'(same-user={r["mean_pairwise_jaccard_same_user"]:.3f}, '
              f'cross-user={r["mean_pairwise_jaccard_cross_user"]:.3f})')
        print(f'  modal set {r["modal_set"]} covers {r["modal_set_share_pct"]}% of samples')
        print(f'  fixed patches (<5% or >95%): {r["n_fixed_patches_(<5%_or_>95%)"]}, '
              f'variable patches: {r["n_variable_patches_(5-95%)"]}')

    overall_mean_jac = np.mean([r['mean_pairwise_jaccard'] for r in per_video.values()])
    overall_modal_share = np.mean([r['modal_set_share_pct'] for r in per_video.values()])
    print(f'\noverall mean pairwise Jaccard (avg over videos): {overall_mean_jac:.3f}')
    print(f'overall modal-set share (avg over videos): {overall_modal_share:.2f}%')

    print('\n=== trajectory feature <-> selection relationships ===')
    rel = analyze_relationships(metas, mask_all, feats, direction, still)
    for k, v in rel.items():
        print(f'  {k}: {v}')

    summary = {
        'checkpoint': CKPT_PATH, 'dataset': 'Jin2022', 'split': 'test', 'num_samples': n,
        'stillness_threshold_deg_per_step': float(still_thresh),
        'direction_distribution': {k: int(v) for k, v in zip(dir_vals, dir_counts)},
        'per_video': per_video,
        'overall_mean_pairwise_jaccard': float(overall_mean_jac),
        'overall_modal_set_share_pct': float(overall_modal_share),
        'relationships': rel,
    }
    with open(os.path.join(OUT_DIR, 'patch_selection_variance_analysis.json'), 'w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f'\nsaved -> {os.path.join(OUT_DIR, "patch_selection_variance_analysis.json")}')

    # persist per-sample records for the visualization script
    records = []
    for i, (video, user, timestep) in enumerate(metas):
        records.append({
            'video': video, 'user': user, 'timestep': timestep,
            'history': histories[i], 'selected_patches': mask_all[i].nonzero()[0].tolist(),
            'logits': logits_all[i], 'mean_speed': float(feats['mean_speed'][i]),
            'direction': direction[i], 'still': bool(still[i]),
            'cur_row': int(feats['cur_row'][i]), 'cur_col': int(feats['cur_col'][i]),
            'cur_patch': int(feats['cur_patch'][i]),
            'dyaw_net': float(feats['dyaw_net'][i]), 'dpitch_net': float(feats['dpitch_net'][i]),
        })
    with open(os.path.join(OUT_DIR, 'patch_selection_variance_records.pkl'), 'wb') as f:
        pickle.dump({'grid_rows': GRID_ROWS, 'grid_cols': GRID_COLS, 'records': records}, f)
    print(f'saved -> {os.path.join(OUT_DIR, "patch_selection_variance_records.pkl")}')


if __name__ == '__main__':
    main()
