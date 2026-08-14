"""
[Task 4] Representative-case visualization: trajectory vs selected patches.

For a handful of hand-picked samples (still / fast-right / fast-left /
fast-down, all drawn from the SAME video so the only thing that differs is
the trajectory), plot:
  top row    - the historical (yaw, pitch) path with an arrow showing net
               movement, over the 4x4 grid boundaries
  bottom row - which of the 16 patches got selected by
               PatchSelectionModule.select_patches() for that sample

Run inside the vp_netllm conda env (after patch_selection_variance_analysis.py
has produced patch_selection_variance_records.pkl):
    conda activate vp_netllm && python analysis/plot_patch_selection_cases.py
"""
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

BLUE_SELECTED = '#256abf'
GRAY_UNSELECTED = '#e7ebf0'
ORANGE_ACCENT = '#eb6834'
INK = '#1a1a1a'
MUTED = '#6b7280'

# (record_index, short label) - chosen from patch_selection_variance_records.pkl,
# all from video 14 so only the trajectory differs between panels. Direction is
# based on the CUMULATIVE (unwrapped) net yaw/pitch displacement over the
# history window, not a naive wrap180(last-first) (see patch_selection_variance_analysis.py
# for why that naive form silently flips sign for large swings).
CASES = [
    (863, 'still\n(speed≈0)'),
    (709, 'fast RIGHT'),
    (600, 'fast LEFT'),
    (728, 'fast DOWN'),
]


def wrap180(deg):
    return (deg + 180.0) % 360.0 - 180.0


def plot_trajectory(ax, history, grid_rows, grid_cols):
    pitch = history[:, 1]
    yaw = history[:, 2]

    # unwrap yaw for plotting a continuous path (avoid a spurious line across the seam)
    yaw_unwrapped = np.copy(yaw)
    for i in range(1, len(yaw_unwrapped)):
        yaw_unwrapped[i] = yaw_unwrapped[i - 1] + wrap180(yaw[i] - yaw[i - 1])

    pad = 20
    x_lo = min(yaw_unwrapped.min(), -180) - pad
    x_hi = max(yaw_unwrapped.max(), 180) + pad

    # column/row boundary grid lines, repeated across the full unwrapped x-range
    import math
    col_bounds = [-180 + c * 360 / grid_cols for c in range(1, grid_cols)]
    k_lo, k_hi = math.floor((x_lo + 180) / 360) - 1, math.ceil((x_hi + 180) / 360) + 1
    for k in range(k_lo, k_hi):
        for cb in col_bounds:
            x = cb + k * 360
            if x_lo <= x <= x_hi:
                ax.axvline(x, color='#d7dce2', linewidth=1, zorder=0)
    for r in range(1, grid_rows):
        ax.axhline(90 - r * 180 / grid_rows, color='#d7dce2', linewidth=1, zorder=0)

    ax.plot(yaw_unwrapped, pitch, color=MUTED, linewidth=1.5, alpha=0.6, zorder=1)
    ax.scatter(yaw_unwrapped[:-1], pitch[:-1], color=MUTED, s=14, zorder=2)
    ax.scatter(yaw_unwrapped[0], pitch[0], color=MUTED, s=40, marker='o',
                edgecolor='white', linewidth=1, zorder=3, label='start')

    # net-displacement arrow (start -> end), matching the direction label's definition
    arrow = FancyArrowPatch((yaw_unwrapped[0], pitch[0]), (yaw_unwrapped[-1], pitch[-1]),
                              arrowstyle='-|>', mutation_scale=18, color=ORANGE_ACCENT,
                              linewidth=2, zorder=4, shrinkA=6, shrinkB=6)
    ax.add_patch(arrow)
    ax.scatter(yaw_unwrapped[-1], pitch[-1], color=ORANGE_ACCENT, s=70, marker='*',
                edgecolor='white', linewidth=0.8, zorder=5, label='current (last)')

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(-100, 100)
    ax.set_xlabel('yaw (°, unwrapped)', fontsize=8, color=MUTED)
    ax.set_ylabel('pitch (°)', fontsize=8, color=MUTED)
    ax.tick_params(labelsize=7)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)


def plot_patch_grid(ax, selected_patches, cur_row, cur_col, grid_rows, grid_cols):
    selected = set(selected_patches)
    for r in range(grid_rows):
        for c in range(grid_cols):
            idx = r * grid_cols + c
            is_sel = idx in selected
            is_cur = (r == cur_row and c == cur_col)
            face = BLUE_SELECTED if is_sel else GRAY_UNSELECTED
            rect = Rectangle((c, grid_rows - 1 - r), 1, 1, facecolor=face,
                               edgecolor='white', linewidth=2)
            ax.add_patch(rect)
            txt_color = 'white' if is_sel else INK
            ax.text(c + 0.5, grid_rows - 1 - r + 0.5, str(idx), ha='center', va='center',
                     fontsize=10, color=txt_color, fontweight='bold')
            if is_cur:
                ax.add_patch(Rectangle((c, grid_rows - 1 - r), 1, 1, facecolor='none',
                                         edgecolor=ORANGE_ACCENT, linewidth=3.5))

    ax.set_xlim(0, grid_cols)
    ax.set_ylim(0, grid_rows)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    with open(os.path.join(OUT_DIR, 'patch_selection_variance_records.pkl'), 'rb') as f:
        d = pickle.load(f)
    recs = d['records']
    grid_rows, grid_cols = d['grid_rows'], d['grid_cols']

    fig, axes = plt.subplots(2, len(CASES), figsize=(4.2 * len(CASES), 8.0), dpi=200)
    fig.patch.set_facecolor('white')

    for col, (idx, label) in enumerate(CASES):
        r = recs[idx]
        ax_top, ax_bot = axes[0, col], axes[1, col]

        plot_trajectory(ax_top, r['history'], grid_rows, grid_cols)
        plot_patch_grid(ax_bot, r['selected_patches'], r['cur_row'], r['cur_col'], grid_rows, grid_cols)

        ax_top.set_title(
            f'{label}\nvideo{r["video"]} user{r["user"]} t={r["timestep"]}\n'
            f'speed={r["mean_speed"]:.2f}°/step',
            fontsize=9.5, color=INK)
        ax_bot.set_xlabel(f'selected: {len(r["selected_patches"])}/16 patches', fontsize=9, color=MUTED)

        if col == 0:
            ax_top.legend(loc='upper left', fontsize=7, frameon=False)

    handles = [
        plt.Line2D([0], [0], marker='s', color='none', markerfacecolor=BLUE_SELECTED, markersize=12, label='selected patch'),
        plt.Line2D([0], [0], marker='s', color='none', markerfacecolor=GRAY_UNSELECTED, markersize=12, label='not selected'),
        plt.Line2D([0], [0], marker='s', color='none', markerfacecolor='none', markeredgecolor=ORANGE_ACCENT,
                    markeredgewidth=2.5, markersize=12, label='current position patch'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('Trajectory vs. selected patches — same video (14), four different histories',
                  fontsize=13, color=INK, y=1.0)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    out_path = os.path.join(OUT_DIR, 'patch_selection_case_studies.png')
    fig.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    print(f'saved -> {out_path}')


if __name__ == '__main__':
    main()
