"""
Generates a two-panel training chart (loss curves + validation metrics)
from a YOLO results.csv file, matching the style used in the README.

Usage: python3 generate_training_chart.py <results.csv> <output.png> ["subtitle text"]
Requires: pandas, matplotlib
"""
import sys
import pandas as pd
import matplotlib.pyplot as plt

results_csv = sys.argv[1]
output_png = sys.argv[2]
subtitle = sys.argv[3] if len(sys.argv) > 3 else "Fire/Smoke YOLOv8s training run"

df = pd.read_csv(results_csv)
df.columns = [c.strip() for c in df.columns]

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.edgecolor': '#3a3a3a',
    'axes.labelcolor': '#2a2a2a',
    'text.color': '#2a2a2a',
    'xtick.color': '#4a4a4a',
    'ytick.color': '#4a4a4a',
    'axes.grid': True,
    'grid.color': '#e0e0e0',
    'grid.linewidth': 0.6,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

ax = axes[0]
loss_colors = {'box': '#d9534f', 'cls': '#e8985e', 'dfl': '#c9a227'}
for key, label in [('box_loss', 'box'), ('cls_loss', 'cls'), ('dfl_loss', 'dfl')]:
    ax.plot(df['epoch'], df[f'train/{key}'], color=loss_colors[label], linewidth=1.8,
            label=f'train {label}')
    ax.plot(df['epoch'], df[f'val/{key}'], color=loss_colors[label], linewidth=1.4,
            linestyle='--', alpha=0.75, label=f'val {label}')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training & Validation Loss', fontsize=13, fontweight='medium', loc='left')
ax.legend(fontsize=8, ncol=2, frameon=False, loc='upper right')
ax.set_xlim(df['epoch'].min(), df['epoch'].max())

ax = axes[1]
metric_colors = {
    'metrics/precision(B)': '#2a6f97', 'metrics/recall(B)': '#5fa8d3',
    'metrics/mAP50(B)': '#2e7d32', 'metrics/mAP50-95(B)': '#81c784',
}
labels = {
    'metrics/precision(B)': 'Precision', 'metrics/recall(B)': 'Recall',
    'metrics/mAP50(B)': 'mAP50', 'metrics/mAP50-95(B)': 'mAP50-95',
}
for key, color in metric_colors.items():
    ax.plot(df['epoch'], df[key], color=color, linewidth=1.9, label=labels[key])
ax.set_xlabel('Epoch')
ax.set_ylabel('Score')
ax.set_title('Validation Metrics', fontsize=13, fontweight='medium', loc='left')
ax.legend(fontsize=9, frameon=False, loc='lower right')
ax.set_xlim(df['epoch'].min(), df['epoch'].max())
ax.set_ylim(0, max(0.8, df['metrics/precision(B)'].max() * 1.15))

for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle(subtitle, fontsize=12, color='#555555', y=1.02)
fig.tight_layout()
fig.savefig(output_png, dpi=180, bbox_inches='tight')
print(f"Saved {output_png}")
