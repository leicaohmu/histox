"""Draw the HistoX workflow overview figure (docs/source/overview.png).

All images are HistoX outputs generated from the open-access TCGA slides
documented in docs/source/_static/slide_processing/generation_metadata.json
and docs/source/methods/multimodal/ps3.rst.

Run from the repository root:  python docs/figures/make_overview.py
"""
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from PIL import Image

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
SRC = ROOT / 'docs/source'
OUT = SRC / 'overview.png'

INK, MUTED, ORANGE, PANEL, LINE = '#262a2e', '#6b7280', '#ee6c4d', '#f5f6f8', '#d6d9de'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})

thumb = Image.open(SRC / '_static/slide_processing/tutorial_roi_tiles.png').convert('RGB')
plain = np.asarray(Image.open(SRC / '_static/slide_processing/tutorial_thumbnail.png').convert('RGB'))
att = Image.open(SRC / '_static/examples/ps3_spatial_attention_tcga_a6_5660.png').convert('RGB')
att = att.crop((55, 40, 385, 525))

fig = plt.figure(figsize=(16, 6.4), dpi=150)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 160); ax.set_ylim(0, 64); ax.axis('off')

def panel(x, w, num, title):
    ax.add_patch(FancyBboxPatch((x, 3), w, 56, boxstyle='round,pad=0,rounding_size=2.2', fc=PANEL, ec='none'))
    ax.add_patch(FancyBboxPatch((x + 2.5, 52.2), 6.2, 4.6, boxstyle='round,pad=0,rounding_size=1.2', fc=ORANGE, ec='none'))
    ax.text(x + 5.6, 54.5, str(num), ha='center', va='center', color='white', fontsize=15, weight='bold')
    ax.text(x + 10.5, 54.5, title, ha='left', va='center', color=INK, fontsize=16, weight='bold')

def image(img, x, y, w, h, border=True):
    ax.imshow(img, extent=(x, x + w, y, y + h), aspect='auto', zorder=3)
    if border:
        ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=LINE, lw=1, zorder=4))

def arrow(x0, y0, x1, y1, color=INK, lw=1.8):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=16, color=color, lw=lw, zorder=5))

def label(x, y, text, **kw):
    kw.setdefault('color', INK); kw.setdefault('fontsize', 11.5)
    ax.text(x, y, text, **kw)

def tile_crops(n, size=46, seed=0):
    """Square crops from tissue-rich regions of the TCGA thumbnail."""
    rng = np.random.default_rng(seed); out = []
    h, w, _ = plain.shape
    while len(out) < n:
        y, x = rng.integers(0, h - size), rng.integers(0, w - size)
        c = plain[y:y + size, x:x + size]
        if (c.mean(axis=2) < 200).mean() > 0.85:
            out.append(c)
    return out

# ---------------------------------------------------------------- phase 1
panel(3, 50, 1, 'Prepare data')
image(thumb.crop((20, 150, 560, 600)), 7, 22, 24, 20)
label(19, 18.5, 'Whole-slide image\n+ ROI and tile grid', ha='center', va='top', fontsize=10.5, color=MUTED)
arrow(32.5, 32, 36.5, 32)
for i, c in enumerate(tile_crops(12, seed=3)):
    r, k = divmod(i, 3)
    image(c, 37.5 + k * 4.9, 39.5 - r * 4.9, 4.4, 4.4, border=False)
label(44.8, 18.5, 'Quality-controlled,\nnormalized tiles', ha='center', va='top', fontsize=10.5, color=MUTED)
label(7, 11.5, '+ tissue and blur QC      + stain normalization', fontsize=10.5, color=INK)
label(7, 7.2, 'Output: tiles or feature bags per slide', fontsize=10.5, color=MUTED)

arrow(53.8, 31, 57.2, 31, color='#9aa0a6', lw=2.4)

# ---------------------------------------------------------------- phase 2
panel(58, 46, 2, 'Train and validate')
for i, c in enumerate(tile_crops(3, seed=11)):
    image(c, 62 + i * 1.2, 36 - i * 1.2, 7, 7)
label(65.5, 30, 'Training folds', ha='center', va='top', fontsize=10.5, color=MUTED)
arrow(71.5, 36, 76, 36)
ax.add_patch(FancyBboxPatch((76.5, 31), 12, 10, boxstyle='round,pad=0,rounding_size=1.6', fc='white', ec=ORANGE, lw=2))
for yy in (38.5, 33.5):
    for xx in (79, 86):
        ax.add_patch(FancyBboxPatch((xx - 1, yy - 0.8), 2, 1.6, boxstyle='round,pad=0,rounding_size=0.5', fc=INK, ec='none'))
ax.plot([80, 82.5, 82.5, 85], [38.5, 38.5, 33.5, 33.5], color=INK, lw=1.4)
label(82.5, 29.6, 'Model', ha='center', va='top', fontsize=11.5, weight='bold')
label(91, 39.5, 'classification\nregression\nsurvival', fontsize=10, color=MUTED, va='top')
arrow(82.5, 26.5, 82.5, 21.5)
label(84.5, 24, 'k-fold validation', fontsize=10.5, color=INK, va='center')
ax.add_patch(Rectangle((77, 8), 11, 11, fc='white', ec=LINE, lw=1))
fpr = np.linspace(0, 1, 50); tpr = 1 - (1 - fpr) ** 3.2
ax.plot(77 + 11 * fpr, 8 + 11 * tpr, color=ORANGE, lw=2)
ax.plot([77, 88], [8, 19], color=LINE, lw=1, ls='--')
label(90, 13.5, 'Metrics\nAUROC, AP,\nC-index', fontsize=10.5, va='center', color=MUTED)

arrow(104.8, 31, 108.2, 31, color='#9aa0a6', lw=2.4)

# ---------------------------------------------------------------- phase 3
panel(109, 48, 3, 'Explain')
image(att, 113, 26, 15, 22)
label(120.5, 24.3, 'Attention or\nprediction heatmaps', ha='center', va='top', fontsize=10.5, color=MUTED)
for i, c in enumerate(tile_crops(16, size=40, seed=21)):
    r, k = divmod(i, 4)
    image(c, 133 + k * 3.9, 44.5 - r * 3.9, 3.6, 3.6, border=False)
label(140.5, 24.3, 'Mosaic map of\nlearned features', ha='center', va='top', fontsize=10.5, color=MUTED)
rng = np.random.default_rng(7)
for cx, cy, col in [(117, 11, ORANGE), (123, 13.5, '#7f2d73'), (120, 7.5, '#4c78a8')]:
    pts = rng.normal(0, 1.1, (25, 2))
    ax.scatter(cx + pts[:, 0], cy + pts[:, 1], s=9, color=col, zorder=4, lw=0)
label(129, 11, 'Feature-space analysis\nand uncertainty', fontsize=10.5, va='center', color=MUTED)

fig.savefig(OUT, dpi=150, facecolor='white')
print('wrote', OUT)
