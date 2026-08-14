"""
Plot the patch-selection frequency heatmap produced by patch_selection_heatmap.py.
Two figures:
  1. patch_selection_heatmap.png       - plain 4x4 grid-coordinate heatmap
  2. patch_selection_heatmap_overlay.png - same heatmap overlaid on a sample
     equirectangular video frame, so grid cells map to actual screen regions.

Run inside the vp_netllm conda env:
    conda activate vp_netllm && python analysis/plot_patch_selection_heatmap.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_FRAME = os.path.join(
    os.path.dirname(OUT_DIR), 'data', 'images', 'Jin2022_images', 'video4_images', '1.jpg')

# dataviz skill sequential-blue ramp (light -> dark), references/palette.md
SEQ_BLUE_HEX = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7',
                 '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']
SEQ_BLUE_CMAP = LinearSegmentedColormap.from_list('seq_blue', SEQ_BLUE_HEX, N=256)

INK_DARK = '#1a1a1a'
INK_LIGHT = '#f5f7fa'
SURFACE_LIGHT = '#ffffff'
GRID_LINE = '#ffffff'


def luminance(hex_color):
    h = hex_color.lstrip('#')
    r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    def lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = lin(r), lin(g), lin(b)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def ink_for(value_pct, vmax):
    color = SEQ_BLUE_CMAP(value_pct / vmax if vmax > 0 else 0.0)
    lum = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]
    return INK_DARK if lum > 0.5 else INK_LIGHT


def load_summary():
    with open(os.path.join(OUT_DIR, 'patch_selection_frequency.json')) as f:
        summary = json.load(f)
    grid = np.load(os.path.join(OUT_DIR, 'patch_selection_freq_grid.npy'))
    return summary, grid


# yaw/pitch band edges implied by utils.patch_labeling.yaw_pitch_to_grid_cell
ROW_LABELS = ['+90° ~ +67.5°\n(top pole)', '+67.5° ~ +22.5°\n(upper)',
              '+22.5° ~ -22.5°\n(horizon)', '-22.5° ~ -90°\n(bottom pole)']
COL_LABELS = ['-180° ~ -90°', '-90° ~ 0°', '0° ~ +90°', '+90° ~ +180°']


def plot_plain(summary, grid):
    rows, cols = grid.shape
    vmax = max(grid.max(), 1.0)

    fig, ax = plt.subplots(figsize=(7.2, 6.6), dpi=200)
    fig.patch.set_facecolor(SURFACE_LIGHT)
    ax.set_facecolor(SURFACE_LIGHT)

    im = ax.imshow(grid, cmap=SEQ_BLUE_CMAP, vmin=0, vmax=vmax, aspect='equal')

    for r in range(rows):
        for c in range(cols):
            v = grid[r, c]
            idx = r * cols + c
            txt = f'{v:.1f}%\n(patch {idx})'
            ax.text(c, r, txt, ha='center', va='center', fontsize=11,
                     color=ink_for(v, vmax), fontweight='bold',
                     path_effects=[pe.withStroke(linewidth=0)])

    ax.set_xticks(range(cols))
    ax.set_xticklabels(COL_LABELS, fontsize=8.5)
    ax.set_yticks(range(rows))
    ax.set_yticklabels(ROW_LABELS, fontsize=8.5)
    ax.set_xlabel('yaw range', fontsize=10, color='#4a4a4a')
    ax.set_ylabel('pitch range', fontsize=10, color='#4a4a4a')

    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which='minor', color=GRID_LINE, linewidth=2)
    ax.tick_params(which='minor', length=0)
    ax.tick_params(which='major', length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)
    cbar.set_label('selection frequency (%)', fontsize=10)
    cbar.outline.set_visible(False)

    n = summary['num_samples']
    ax.set_title(
        f'PatchSelectionModule selection frequency — Jin2022 test split (n={n})\n'
        f'checkpoint: best_patch_selection.pth, threshold={summary["effective_threshold"]} '
        f'(fallback: force top-1 if none pass)',
        fontsize=10.5, color='#2a2a2a', pad=14)

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, 'patch_selection_heatmap.png')
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'saved -> {out_path}')


def plot_overlay(summary, grid):
    if not os.path.exists(SAMPLE_FRAME):
        print(f'sample frame not found at {SAMPLE_FRAME}, skipping overlay')
        return

    img = Image.open(SAMPLE_FRAME).convert('RGB')
    w, h = img.size
    rows, cols = grid.shape
    vmax = max(grid.max(), 1.0)

    fig, ax = plt.subplots(figsize=(9.6, 5.6), dpi=200)
    fig.patch.set_facecolor(SURFACE_LIGHT)
    ax.imshow(img, extent=(0, w, h, 0))

    overlay = ax.imshow(grid, cmap=SEQ_BLUE_CMAP, vmin=0, vmax=vmax,
                          extent=(0, w, h, 0), alpha=0.55, aspect='auto')

    ph, pw = h / rows, w / cols
    for r in range(rows):
        for c in range(cols):
            v = grid[r, c]
            idx = r * cols + c
            cx, cy = (c + 0.5) * pw, (r + 0.5) * ph
            ax.text(cx, cy, f'{v:.0f}%', ha='center', va='center', fontsize=13,
                     color=ink_for(v, vmax), fontweight='bold')
            ax.text(cx, cy + ph * 0.22, f'patch {idx}', ha='center', va='center',
                     fontsize=8, color=ink_for(v, vmax), alpha=0.85)

    for r in range(1, rows):
        ax.axhline(r * ph, color=GRID_LINE, linewidth=1.4, alpha=0.85)
    for c in range(1, cols):
        ax.axvline(c * pw, color=GRID_LINE, linewidth=1.4, alpha=0.85)

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(overlay, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label('selection frequency (%)', fontsize=10)
    cbar.outline.set_visible(False)

    n = summary['num_samples']
    ax.set_title(
        f'Patch selection frequency overlaid on a sample equirectangular frame '
        f'(video4, frame 1) — n={n} test samples',
        fontsize=10.5, color='#2a2a2a', pad=10)

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, 'patch_selection_heatmap_overlay.png')
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'saved -> {out_path}')


if __name__ == '__main__':
    summary, grid = load_summary()
    plot_plain(summary, grid)
    plot_overlay(summary, grid)
