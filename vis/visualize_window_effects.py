"""
visualize_window_effects.py — Show how window size affects logsig features on Epilepsy EEG.

For a seizure and non-seizure example:
  Row 1: raw signal  (+ shaded windows at win64 / win128 scale)
  Row 2: logsig level-1 (net displacement per window) at win64 vs win128
  Row 3: logsig level-2 (Lévy area per window)       at win64 vs win128

The area term is what the model relies on for local geometric discrimination.
Smaller windows → more oscillation per feature; larger windows → cancellation.

Usage: python visualize_window_effects.py
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

from src.dataloader import get_logsig

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

with open('preprocessed_data/_DA_Epilepsy_256_00.pkl', 'rb') as f:
    data = pickle.load(f)

X_raw = torch.tensor(data[0]).transpose(1, 2).float()   # [N, 256, 1]
y     = data[3].astype(int)

# Pick one clear example per class
idx0 = np.where(y == 0)[0][0]   # non-seizure
idx1 = np.where(y == 1)[0][0]   # seizure
examples = [(idx0, 'Non-seizure (class 0)'), (idx1, 'Seizure (class 1)')]

# ---------------------------------------------------------------------------
# Compute sliding-window logsig features
# depth=2 on 1-channel + time augmentation → 3 features:
#   [0] level-1 time (always ~1 per window, mostly constant)
#   [1] level-1 signal (net displacement = last − first in window)
#   [2] level-2 area (Lévy area swept by the path)
# ---------------------------------------------------------------------------

WIN_SIZES  = [64, 128]
WIN_COLORS = {64: '#DD8452', 128: '#4C72B0'}
WIN_LABELS = {64: 'win 64', 128: 'win 128'}

logsigs = {}
for ws in WIN_SIZES:
    ls = get_logsig(X_raw, depth=2, mode='window', window_size=ws)   # [N, 256, 3]
    logsigs[ws] = ls.numpy()

T = X_raw.shape[1]
t = np.arange(T)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True)
fig.suptitle(
    'Epilepsy EEG — logsig features at different window scales',
    fontsize=11, fontweight='bold', y=1.01,
)

ROW_LABELS = [
    'Raw EEG signal',
    'Level-1: net displacement\nper window',
    'Level-2: Lévy area\nper window',
]

for col, (idx, title) in enumerate(examples):
    sig  = X_raw[idx, :, 0].numpy()   # [256]

    for row in range(3):
        ax = axes[row, col]

        if row == 0:
            ax.plot(t, sig, color='#333', linewidth=0.8, zorder=3)

            # shade one exemplar window at each size near the middle of the signal
            anchor = T // 2
            for ws in WIN_SIZES:
                start = max(0, anchor - ws)
                ax.axvspan(start, anchor, alpha=0.18,
                           color=WIN_COLORS[ws], zorder=2,
                           label=WIN_LABELS[ws])
            ax.set_ylabel(ROW_LABELS[row], fontsize=8.5)
            if col == 0:
                ax.legend(fontsize=8, frameon=False, loc='upper left')

        else:
            feat_idx = row - 1   # row 1 → feat 1 (level-1 signal), row 2 → feat 2 (area)
            for ws in WIN_SIZES:
                feat = logsigs[ws][idx, :, feat_idx + 1]   # skip feat[0] (time)
                ax.plot(t, feat, color=WIN_COLORS[ws],
                        linewidth=0.9, alpha=0.85,
                        label=WIN_LABELS[ws])

            ax.axhline(0, color='#aaa', linewidth=0.5, zorder=1)
            ax.set_ylabel(ROW_LABELS[row], fontsize=8.5)
            if row == 1 and col == 0:
                ax.legend(fontsize=8, frameon=False, loc='upper left')

        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', linestyle='--', alpha=0.3)

        if row == 0:
            ax.set_title(title, fontsize=10, pad=6)

    axes[2, col].set_xlabel('Timestep', fontsize=9)

# Shared y-limits for rows 1 and 2 so amplitudes are comparable across classes
for row in [1, 2]:
    all_vals = np.concatenate([
        logsigs[ws][idx, :, row] for ws in WIN_SIZES for idx, _ in examples
    ])
    vmax = np.percentile(np.abs(all_vals), 98) * 1.1
    for col in range(2):
        axes[row, col].set_ylim(-vmax, vmax)

plt.tight_layout()
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/window_effects.pdf', bbox_inches='tight')
plt.savefig('plots/window_effects.png', dpi=150, bbox_inches='tight')
print('Saved: plots/window_effects.pdf  plots/window_effects.png')
plt.show()
