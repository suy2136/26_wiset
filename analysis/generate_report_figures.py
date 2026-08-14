"""
Generates the 5 report figures for the patch-selection integration project
summary, into /workspace/report_assets/. Uses only already-measured numbers
from this session's logs/reports -- no new computation/training.
"""
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

OUT_DIR = '/workspace/report_assets'
os.makedirs(OUT_DIR, exist_ok=True)

# --- Korean font setup ---
font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150

COLORS = {
    'baseline': '#4C72B0',
    'all-patch': '#DD8452',
    'patch-selection': '#55A868',
}

# ============================================================
# 1. 3-condition MAE comparison, grouped bar, 4 experiments
# ============================================================
experiments = ['원래 4epoch\n(LR 부스트 없음)', '14분 방향확인\n(1000샘플/1ep/LR×5)', 'LR×5\n(2ep/전체데이터)', 'LR×2\n(결과 대기 중)']
data = {
    'baseline': [20.76, 41.53, 21.87, None],
    'all-patch': [23.92, 29.29, 40.55, None],
    'patch-selection': [54.72, 52.72, 49.69, None],
}

fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(experiments))
width = 0.25
for i, (mode, vals) in enumerate(data.items()):
    plot_vals = [v if v is not None else 0 for v in vals]
    bars = ax.bar(x + (i - 1) * width, plot_vals, width, label=mode, color=COLORS[mode])
    for j, (bar, v) in enumerate(zip(bars, vals)):
        if v is not None:
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.8, f'{v:.1f}°', ha='center', fontsize=9)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, 2, '대기 중', ha='center', fontsize=8,
                     rotation=90, color='gray', style='italic')
ax.set_xticks(x)
ax.set_xticklabels(experiments)
ax.set_ylabel('MAE (도, rotation-aware)')
ax.set_title('3-Condition VP MAE 비교 — 4개 실험')
ax.legend(title='multimodal_mode')
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'mae_comparison_4experiments.png'))
plt.close(fig)

# ============================================================
# 2. patch-selection vs all-patch gap trend, line chart
# ============================================================
gap_labels = ['원래\n(4ep, no boost)', '14분 방향확인\n(1000/1ep/LR×5)', 'LR×5\n(2ep/전체)', 'LR×2\n(대기 중)']
gap_pct = [128.7, 80.0, 22.5, None]

fig, ax = plt.subplots(figsize=(9, 5.5))
xs = np.arange(len(gap_labels))
known = [(i, v) for i, v in enumerate(gap_pct) if v is not None]
ax.plot([i for i, v in known], [v for i, v in known], marker='o', markersize=9, linewidth=2.5, color='#C44E52')
for i, v in known:
    ax.annotate(f'{v:.1f}%', (i, v), textcoords='offset points', xytext=(0, 12), ha='center', fontsize=11, fontweight='bold')
ax.axhline(0, color='gray', linewidth=0.8)
ax.set_xticks(xs)
ax.set_xticklabels(gap_labels)
ax.set_ylabel('patch-selection이 all-patch보다 나쁜 정도 (%)')
ax.set_title('patch-selection ↔ all-patch MAE 격차 추이\n(주의: LR×5 시점 격차 축소는 all-patch 자체 악화로 인한 confound 있음 — 본문 참조)')
ax.grid(axis='y', alpha=0.3)
ax.text(2, 22.5 + 8, '[주의] all-patch 자체가\n70% 악화된 상태의 격차', fontsize=9, color='#C44E52',
        ha='center', style='italic')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'gap_trend_patchselection_vs_allpatch.png'))
plt.close(fig)

# ============================================================
# 3. Selector+Speculative ablation (A/B/C/D), MAE + latency side by side
# ============================================================
configs = ['A. direct\n(no wrapper)', 'B. Selector\nRecentK(k=6)', 'C. Speculative\n(γ=4,th=0.3)', 'D. Selector+\nSpeculative']
mae_vals = [20.7582, 20.5011, 20.8754, 20.6207]
fc_vals = [20.00, 20.00, 6.58, 6.50]
lat_vals = [413.22, 403.76, 173.49, 163.35]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
bars1 = ax1.bar(configs, mae_vals, color=['#8C8C8C', '#4C72B0', '#DD8452', '#55A868'])
for b, v in zip(bars1, mae_vals):
    ax1.text(b.get_x() + b.get_width() / 2, v + 0.15, f'{v:.2f}°', ha='center', fontsize=10)
ax1.set_ylabel('MAE (도)')
ax1.set_title('MAE (baseline 모드, 전체 1,698샘플)')
ax1.set_ylim(0, max(mae_vals) * 1.25)
ax1.grid(axis='y', alpha=0.3)

bars2 = ax2.bar(configs, lat_vals, color=['#8C8C8C', '#4C72B0', '#DD8452', '#55A868'])
for b, v, fc in zip(bars2, lat_vals, fc_vals):
    ax2.text(b.get_x() + b.get_width() / 2, v + 8, f'{v:.0f}ms\n(fc={fc:.1f})', ha='center', fontsize=9)
ax2.set_ylabel('Latency (ms/sample)')
ax2.set_title('Latency + forward count (fc)')
ax2.set_ylim(0, max(lat_vals) * 1.3)
ax2.grid(axis='y', alpha=0.3)

fig.suptitle('Selector + Speculative Decoding Ablation (D: 최종 통합 구성)', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'selector_speculative_ablation.png'))
plt.close(fig)

# ============================================================
# 4. Patch selection heatmap (reuse existing frequency grid if present)
# ============================================================
freq_grid_path = '/workspace/VP_extract/NetLLM/viewport_prediction/analysis/patch_selection_freq_grid.npy'
if os.path.exists(freq_grid_path):
    grid = np.load(freq_grid_path)  # values are already percentages (0-100), not fractions
    fig, ax = plt.subplots(figsize=(6.5, 6))
    im = ax.imshow(grid, cmap='viridis', vmin=0, vmax=100)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            val = grid[i, j]
            color = 'white' if val < 60 else 'black'
            ax.text(j, i, f'{val:.0f}%', ha='center', va='center', color=color, fontsize=11)
    ax.set_xticks(range(grid.shape[1]))
    ax.set_yticks(range(grid.shape[0]))
    ax.set_title('Patch Selection 빈도 히트맵 (4×4 grid, 사전학습 검증 시점,\n전체 1,698 테스트 샘플 기준, 7/28 측정)')
    fig.colorbar(im, ax=ax, label='선택 빈도 (%)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'patch_selection_heatmap_reused.png'))
    plt.close(fig)
    heatmap_status = 'patch_selection_freq_grid.npy 재사용 (사전학습 검증 시점 데이터, 7/28 -- 통합 후 재생성 아님)'
else:
    heatmap_status = 'MISSING -- patch_selection_freq_grid.npy 없음, 생성 못함'

# ============================================================
# 5. Training loss vs actual MAE divergence (all-patch LR x5 run)
# ============================================================
epochs_x = [0.5, 1.0, 1.5, 2.0]
valid_loss = [0.26901700331865597, 0.14190157831102523, 0.1575478532578477, 0.1375240177976106]
# only the final checkpoint's true MAE was measured (valid=38.14, test=40.55);
# intermediate points are not independently measured (checkpoint dir was
# overwritten each validation) -- shown as a single confirmed endpoint,
# annotated clearly as such, not interpolated as if measured.

fig, ax1 = plt.subplots(figsize=(9.5, 6))
ax1.plot(epochs_x, valid_loss, marker='o', color='#4C72B0', linewidth=2.5, label='Valid loss (정규화 Tanh-space MSE, 학습 중 기록)')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Valid loss (정규화 MSE)', color='#4C72B0')
ax1.tick_params(axis='y', labelcolor='#4C72B0')
ax1.set_ylim(0, 0.32)

ax2 = ax1.twinx()
ax2.scatter([2.0], [38.14], color='#C44E52', s=140, zorder=5, marker='D',
            label='실측 rotation-aware MAE (degree, valid set)')
ax2.annotate('실측: 38.14°\n(test set 40.55°와 거의 동일)', (2.0, 38.14), textcoords='offset points',
             xytext=(-140, -10), fontsize=10, color='#C44E52', fontweight='bold')
ax2.set_ylabel('실제 MAE (도, rotation-aware)', color='#C44E52')
ax2.tick_params(axis='y', labelcolor='#C44E52')
ax2.set_ylim(0, 60)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', fontsize=9)
ax1.set_title('학습 지표(Valid loss) vs 실제 MAE의 괴리\n(all-patch, LR×5 2-epoch 실행)')
ax1.grid(alpha=0.3)
fig.text(0.5, 0.01, '※ epoch 0.5~1.5 지점의 실제 MAE는 측정되지 않음(checkpoint가 매 검증마다 덮어써짐) — 최종(epoch 2.0) 지점만 실측 확인',
          ha='center', fontsize=8, style='italic', color='gray')
fig.tight_layout(rect=[0, 0.03, 1, 1])
fig.savefig(os.path.join(OUT_DIR, 'loss_vs_mae_divergence.png'))
plt.close(fig)

print('Generated figures:')
for f in sorted(os.listdir(OUT_DIR)):
    if f.endswith('.png'):
        print(' ', f)
print(f'Heatmap status: {heatmap_status}')
