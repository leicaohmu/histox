"""Draw the cell-segmentation workflow diagram (docs/source/cell_segmentation.png).

Line-art schematic matching docs/figures/make_overview.py.
Run from the repository root:  python docs/figures/make_cellseg_diagram.py
"""
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
OUT = ROOT / 'docs/source/cell_segmentation.png'
FONT = 'Inter' if any('Inter' in f.name for f in font_manager.fontManager.ttflist) else 'DejaVu Sans'
plt.rcParams.update({'font.family': FONT})
INK = '#111111'

fig = plt.figure(figsize=(14, 3.4), dpi=150)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 140); ax.set_ylim(0, 34); ax.axis('off')

rows = [('One slide', 'Tune segmentation\nparameters', 'Cellpose model and cell diameter'),
        ('All slides', 'Segment cells and\nfind centroids', 'Masks saved per slide'),
        ('All slides', 'Extract cell images\nat centroids', 'With or without the mask'),
        ('Output', 'Cell-level images', 'Loose images or TFRecords')]
w, gap, x = 29, 6, 2.5
for i, (scope, title, body) in enumerate(rows):
    ax.text(x, 29, scope.upper(), fontsize=10.5, color='#555555', weight='bold', va='center')
    ax.add_patch(Rectangle((x, 7), w, 18, fill=False, ec=INK, lw=1.6))
    ax.text(x + w / 2, 18.6, title, fontsize=13, weight='bold', color=INK, ha='center', va='center', linespacing=1.3)
    ax.text(x + w / 2, 10.8, body, fontsize=11.5, color=INK, ha='center', va='center')
    ax.text(x, 1.5, str(i + 1) if i < 3 else '', fontsize=22, weight='bold', color=INK)
    if i < len(rows) - 1:
        ax.add_patch(FancyArrowPatch((x + w + 0.8, 16), (x + w + gap - 0.8, 16), arrowstyle='-|>',
                                     mutation_scale=18, color=INK, lw=2.2, shrinkA=0, shrinkB=0))
    x += w + gap
fig.savefig(OUT, dpi=150, facecolor='white')
print('wrote', OUT)
