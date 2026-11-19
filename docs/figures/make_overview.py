"""Draw the HistoX workflow overview figure (docs/source/overview.png).

Line-art schematic in three phases. All embedded images are HistoX outputs
generated from open-access TCGA slides (see the generation metadata next to
each source image in docs/source).

Run from the repository root:  python docs/figures/make_overview.py
"""
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle, FancyBboxPatch
from PIL import Image

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
SRC = ROOT / 'docs/source'
OUT = SRC / 'overview.png'

FONT = 'Inter' if any('Inter' in f.name for f in font_manager.fontManager.ttflist) else 'DejaVu Sans'
plt.rcParams.update({'font.family': FONT})
INK, GREY = '#111111', '#9a9a9a'

roi = Image.open(SRC / '_static/slide_processing/tutorial_roi_tiles.png').convert('RGB').crop((20, 150, 600, 610))
plain = np.asarray(Image.open(SRC / '_static/slide_processing/tutorial_thumbnail.png').convert('RGB'))
att = Image.open(SRC / '_static/examples/ps3_spatial_attention_tcga_a6_5660.png').convert('RGB').crop((55, 40, 385, 525))
mosaic = Image.open(SRC / 'mosaic_example.png').convert('RGB')
mosaic = mosaic.crop((60, 640, 1176, 1150))
umap = Image.open(SRC / 'umap_example.png').convert('RGB').crop((105, 40, 800, 880))   # points only, no legend

fig = plt.figure(figsize=(16, 7.2), dpi=150)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 160); ax.set_ylim(0, 72); ax.axis('off')


def img(im, x, y, w, h, frame=True, lw=1.2):
    ax.imshow(im, extent=(x, x + w, y, y + h), aspect='auto', zorder=2)
    if frame:
        ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=INK, lw=lw, zorder=3))


def arrow(x0, y0, x1, y1, color=INK, lw=2.2, ms=18):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=ms,
                                 color=color, lw=lw, zorder=4, shrinkA=0, shrinkB=0))


def text(x, y, s, size=13, weight='normal', **kw):
    ax.text(x, y, s, fontsize=size, weight=weight, color=INK, **kw)


def bag(cx, cy, w=7.0, h=7.6, rows=6):
    """Feature-bag icon: a matrix whose rows are tile embeddings."""
    x0, y0 = cx - w / 2, cy - h / 2
    rng = np.random.default_rng(int(cx * 10 + cy))
    rh = h / rows
    for r in range(rows):
        for c in range(5):
            shade = 0.25 + 0.65 * rng.random()
            ax.add_patch(Rectangle((x0 + c * w / 5, y0 + r * rh), w / 5, rh,
                                   fc=str(shade), ec='white', lw=0.8, zorder=2))
    ax.add_patch(Rectangle((x0, y0), w, h, fill=False, ec=INK, lw=1.4, zorder=3))


def model(cx, cy, r=6.2):
    """Model icon: a small fully connected network inside a circle."""
    ax.add_patch(Circle((cx, cy), r, fill=False, ec=INK, lw=2.2, zorder=3))
    layers = [[-2.2, 0, 2.2], [-3.0, -1.0, 1.0, 3.0], [-1.1, 1.1]]
    xs = [cx - 3.6, cx, cx + 3.6]
    pts = [[(x, cy + dy) for dy in ys] for x, ys in zip(xs, layers)]
    for a, b in zip(pts[:-1], pts[1:]):
        for p in a:
            for q in b:
                ax.plot([p[0], q[0]], [p[1], q[1]], color=INK, lw=0.6, zorder=3)
    for layer in pts:
        for (x, y) in layer:
            ax.add_patch(Circle((x, y), 0.65, fc='white', ec=INK, lw=1.3, zorder=4))


def roc(x, y, s=10):
    ax.add_patch(Rectangle((x, y), s, s, fc='white', ec=INK, lw=1.4, zorder=2))
    f = np.linspace(0, 1, 60)
    ax.fill_between(x + s * f, y + s * f, y + s * (1 - (1 - f) ** 3.5), color='#e6e6e6', zorder=2)
    ax.plot(x + s * f, y + s * (1 - (1 - f) ** 3.5), color=INK, lw=1.8, zorder=3)
    ax.plot([x, x + s], [y, y + s], color=GREY, lw=1, ls=(0, (2, 2)), zorder=3)
    text(x + s - 0.6, y + 0.8, 'ROC', size=10, ha='right')


def tiles(x, y, cols, rows, size=4.2, gap=0.7, seed=3):
    rng = np.random.default_rng(seed); h, w, _ = plain.shape; k = 48; n = 0
    while n < cols * rows:
        yy, xx = rng.integers(0, h - k), rng.integers(0, w - k)
        c = plain[yy:yy + k, xx:xx + k]
        if (c.mean(axis=2) < 200).mean() > 0.85:
            r, q = divmod(n, cols)
            img(c, x + q * (size + gap), y + (rows - 1 - r) * (size + gap), size, size, lw=0.8)
            n += 1


# ---------------------------------------------------------------- 1. data
D = 6   # horizontal offset of phase 1
img(roi, 7 + D, 50, 24, 19, lw=1.6)
arrow(19 + D, 48.5, 19 + D, 39.5)
text(17.5 + D, 44, 'Tile extraction', size=15, ha='right', va='center')
text(21 + D, 44, '+ Tissue detection\n+ Blur QC\n+ Stain normalization', size=11.5, va='center', linespacing=1.45)
tiles(7.2 + D, 22.5, 5, 3)
arrow(19 + D, 20.5, 19 + D, 13.5)
text(17.5 + D, 17, 'Feature extraction', size=15, ha='right', va='center')
bag(19 + D, 8.3)
text(19 + D, 2.6, 'Feature bags', size=12.5, weight='bold', ha='center')

# ---------------------------------------------------------------- 2. model
for x in (66, 76, 86):
    bag(x, 64)
arrow(76, 58.5, 76, 49)
text(74.5, 53.7, 'Training', size=15, ha='right', va='center')
text(78, 53.7, 'Classification,\nregression,\nor survival', size=11.5, va='center', linespacing=1.4)
model(76, 40.5)
text(76, 31.4, 'Model', size=12.5, weight='bold', ha='center')
for x in (90, 99):
    bag(x, 40.5)
ax.plot([76, 76, 94.5, 94.5], [29, 26, 26, 35.5], color=INK, lw=2.2, zorder=3, solid_joinstyle='miter')
arrow(85.25, 26, 85.25, 18.5)
text(83.75, 22.3, 'Validation', size=15, ha='right', va='center')
roc(80.25, 6.5)
text(85.25, 2.6, 'Metrics', size=12.5, weight='bold', ha='center')

# ---------------------------------------------------------------- 3. evidence
model(121, 61.5, r=5.6)
text(121, 52.7, 'Model', size=12.5, weight='bold', ha='center')
for x in (134, 143):
    bag(x, 61.5)
ax.plot([121, 121, 138.5, 138.5], [50.6, 48, 48, 57.2], color=INK, lw=2.2, zorder=3, solid_joinstyle='miter')
arrow(129.75, 48, 129.75, 42.5)
img(att, 113, 26.5, 10.5, 15)
text(125.5, 34, 'Attention and\nprediction heatmaps', size=12.5, va='center', linespacing=1.35)
img(mosaic, 113, 13.5, 18, 8.3)
text(133.5, 17.6, 'Mosaic maps', size=12.5, va='center')
img(umap, 113, 1.2, 9, 10)
text(124.5, 6.2, 'Feature-space\nanalysis', size=12.5, va='center', linespacing=1.35)

# ---------------------------------------------------------------- phases
for x, n in ((1.5, '1'), (58, '2'), (109, '3')):
    text(x, 1.4, n, size=30, weight='bold', va='bottom')
arrow(50, 40.5, 56, 40.5, color=GREY, lw=3.2, ms=22)
arrow(104.5, 40.5, 110, 40.5, color=GREY, lw=3.2, ms=22)

fig.savefig(OUT, dpi=150, facecolor='white')
print('wrote', OUT)
