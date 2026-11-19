"""Draw the cell-segmentation workflow diagram (docs/source/cell_segmentation.png).

Schematic only. Run from the repository root:
    python docs/figures/make_cellseg_diagram.py
"""
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
OUT = ROOT / 'docs/source/cell_segmentation.png'
INK, MUTED, ORANGE, PANEL = '#262a2e', '#6b7280', '#ee6c4d', '#f5f6f8'
plt.rcParams.update({'font.family': 'DejaVu Sans'})

fig = plt.figure(figsize=(14, 3.6), dpi=150)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 140); ax.set_ylim(0, 36); ax.axis('off')

steps = [
    ('One slide', 'Tune parameters', 'Choose the Cellpose model and\ncell diameter interactively'),
    ('All slides', 'Segment cells', 'Save masks and cell centroids\nfor every slide'),
    ('All slides', 'Extract cell images', 'Crop a tile at each centroid,\nwith or without the mask'),
    ('Output', 'Cell-level data', 'Loose images or TFRecords\nfor training and analysis'),
]
w, gap, x = 30.5, 4.5, 3
for i, (scope, title, body) in enumerate(steps):
    ax.add_patch(FancyBboxPatch((x, 4), w, 28, boxstyle='round,pad=0,rounding_size=2', fc=PANEL, ec='none'))
    ax.add_patch(FancyBboxPatch((x + 2.2, 24.6), 5.2, 4.6, boxstyle='round,pad=0,rounding_size=1.1',
                                fc=ORANGE if i < 3 else INK, ec='none'))
    ax.text(x + 4.8, 26.9, str(i + 1) if i < 3 else '✓', ha='center', va='center', color='white', fontsize=13, weight='bold')
    ax.text(x + 9.2, 26.9, scope, ha='left', va='center', color=MUTED, fontsize=11)
    ax.text(x + 2.4, 18.5, title, ha='left', va='center', color=INK, fontsize=14.5, weight='bold')
    ax.text(x + 2.4, 11.5, body, ha='left', va='center', color=INK, fontsize=10.8, linespacing=1.5)
    if i < len(steps) - 1:
        ax.add_patch(FancyArrowPatch((x + w + 0.6, 18), (x + w + gap - 0.6, 18), arrowstyle='-|>',
                                     mutation_scale=16, color='#9aa0a6', lw=2.2))
    x += w + gap
fig.savefig(OUT, dpi=150, facecolor='white')
print('wrote', OUT)
