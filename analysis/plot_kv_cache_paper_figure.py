"""
Renders the paper figure comparing patch-selection latency without vs with the KV
cache, from analysis/kv_cache_paper_figure_results.json (produced by
analysis/benchmark_kv_cache_paper_figure.py). Two-bar comparison chart following the
dataviz skill's mark specs: fixed categorical slot order (blue=before, orange=after),
thin bars, direct value + % labels, no gridlines/junk, single axis.
"""
import json
import os

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

_KO_FONT_PATH = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
if os.path.exists(_KO_FONT_PATH):
    fm.fontManager.addfont(_KO_FONT_PATH)
    plt.rcParams['font.family'] = fm.FontProperties(fname=_KO_FONT_PATH).get_name()
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, 'kv_cache_paper_figure_results.json')) as f:
    R = json.load(f)

# validated categorical slots (references/palette.md): slot 1 blue = "before", slot 2 orange = "after"
COLOR_BEFORE = '#2a78d6'
COLOR_AFTER = '#eb6834'
INK_PRIMARY = '#0b0b0b'
INK_SECONDARY = '#52514e'
INK_MUTED = '#898781'
BASELINE = '#c3c2b7'
SUCCESS = '#006300'
SURFACE = '#fcfcfb'

labels = ['KV cache 적용 전\n(patch-selection)', 'KV cache 적용 후\n(최종 patch-selection)']
means = [R['no_kv_cache_ms_avg'], R['with_kv_cache_ms_avg']]
trials = [R['no_kv_cache_ms_per_trial'], R['with_kv_cache_ms_per_trial']]
mins = [min(t) for t in trials]
maxs = [max(t) for t in trials]
colors = [COLOR_BEFORE, COLOR_AFTER]

fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

x = np.arange(2)
bar_width = 0.5
bars = ax.bar(x, means, width=bar_width, color=colors, zorder=3,
              edgecolor='none')

# individual-trial range as a thin error bar (min-max across N_TRIALS), not std,
# since N_TRIALS=3 is too small for a meaningful std whisker
err_low = [means[i] - mins[i] for i in range(2)]
err_high = [maxs[i] - means[i] for i in range(2)]
ax.errorbar(x, means, yerr=[err_low, err_high], fmt='none', ecolor=INK_MUTED,
            elinewidth=1.3, capsize=5, capthick=1.3, zorder=4)

# direct value labels above each bar
for xi, m in zip(x, means):
    ax.text(xi, m + 6, f'{m:.1f}ms', ha='center', va='bottom',
            fontsize=13, fontweight='bold', color=INK_PRIMARY)

# improvement annotation between the bars
mid_y = (means[0] + means[1]) / 2
ax.annotate('', xy=(1 - bar_width / 2 - 0.03, means[1] + 10), xytext=(0 + bar_width / 2 + 0.03, means[0] - 10),
            arrowprops=dict(arrowstyle='-|>', color=SUCCESS, lw=1.6))
ax.text(0.5, mid_y + 2,
        f"-{R['absolute_saving_ms']:.1f}ms\n(-{R['improvement_pct']:.1f}%)",
        ha='center', va='center', fontsize=12.5, fontweight='bold', color=SUCCESS,
        bbox=dict(boxstyle='round,pad=0.35', facecolor=SURFACE, edgecolor=SUCCESS, linewidth=1.2))

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11, color=INK_PRIMARY)
ax.set_ylabel('Inference latency (ms, mean of 10 runs)', fontsize=11, color=INK_SECONDARY)
ax.set_ylim(0, max(means) * 1.28)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color(BASELINE)
ax.tick_params(axis='y', colors=INK_MUTED, labelsize=9.5)
ax.tick_params(axis='x', length=0)
ax.yaxis.grid(True, color='#e1e0d9', linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

fig.suptitle('KV 캐시 적용 전 vs 최종 patch-selection 추론 지연시간',
             fontsize=13.5, fontweight='bold', color=INK_PRIMARY, y=0.98)
ax.set_title(f"RTX 5090 · llama-7b (fp16) · batch=1 · {R['n_trials']} trials × {R['iters']} iters "
             f"(warmup {R['warmup']}) · error bar = trial min-max",
             fontsize=9, color=INK_MUTED, pad=12)

fig.tight_layout(rect=[0, 0, 1, 0.94])
out_path = os.path.join(HERE, 'kv_cache_before_after.png')
fig.savefig(out_path, facecolor=fig.get_facecolor())
print(f'wrote {out_path}')
