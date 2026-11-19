"""Draw the HistoX workflow overview figure (docs/source/overview.png).

Three-phase line-art schematic. Embedded images are HistoX outputs generated
from open-access TCGA slides (see the generation metadata in docs/source).

Run from the repository root:  python docs/figures/make_overview.py
"""
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.path import Path as MPath
from matplotlib.patches import (FancyArrowPatch, FancyBboxPatch, Rectangle, Circle,
                                Ellipse, PathPatch, Polygon)
from PIL import Image

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
SRC = ROOT / 'docs/source'
OUT = SRC / 'overview.png'

FONT = 'Inter' if any('Inter' in f.name for f in font_manager.fontManager.ttflist) else 'DejaVu Sans'
plt.rcParams.update({'font.family': FONT})
INK, ARROW_GREY, FILL, RED = '#000000', '#a6a6a6', '#e8e8e8', '#d7191c'
LW = 2.4

thumb = Image.open(SRC / '_static/slide_processing/tutorial_thumbnail.png').convert('RGB')
plain = np.asarray(thumb)
att = np.asarray(Image.open(SRC / '_static/examples/ps3_spatial_attention_tcga_a6_5660.png')
                 .convert('RGB').crop((55, 40, 385, 525))).copy()
_hsv = np.asarray(Image.fromarray(att).convert('HSV'))
att[_hsv[..., 1] < 40] = 255            # slide background -> white, keep the attention overlay
mosaic = Image.open(SRC / 'mosaic_example.png').convert('RGB').crop((60, 600, 1176, 1150))
umap = Image.open(SRC / 'umap_example.png').convert('RGB').crop((105, 560, 1230, 885))   # lower cluster band

fig = plt.figure(figsize=(16, 7.6), dpi=150)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 160); ax.set_ylim(0, 76); ax.set_aspect('equal'); ax.axis('off')


def text(x, y, s, size=17, weight='normal', **kw):
    kw.setdefault('va', 'center')
    ax.text(x, y, s, fontsize=size, weight=weight, color=INK, **kw)


def arrow(x0, y0, x1, y1, color=INK, lw=LW, ms=20):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=ms,
                                 color=color, lw=lw, shrinkA=0, shrinkB=0, zorder=4))


def framed(im, x, y, w, h, radius=1.2, lw=LW, pad=0.6):
    """Image inside a rounded frame."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={radius}',
                                fc='white', ec='none', zorder=1))
    ax.imshow(im, extent=(x + pad, x + w - pad, y + pad, y + h - pad), aspect='auto', zorder=2)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={radius}',
                                fc='none', ec=INK, lw=lw, zorder=3))


def store(cx, cy, w=7.2, h=8.0):
    """Data icon: a cylinder (feature store)."""
    e = h * 0.22
    x0, x1, top, bot = cx - w / 2, cx + w / 2, cy + h / 2 - e / 2, cy - h / 2 + e / 2
    ax.add_patch(Rectangle((x0, bot), w, top - bot, fc=FILL, ec='none', zorder=2))
    ax.add_patch(Ellipse((cx, bot), w, e, fc=FILL, ec=INK, lw=LW, zorder=2))
    ax.add_patch(Rectangle((x0, bot), w, top - bot, fc=FILL, ec='none', zorder=2.5))
    ax.plot([x0, x0], [bot, top], color=INK, lw=LW, zorder=3)
    ax.plot([x1, x1], [bot, top], color=INK, lw=LW, zorder=3)
    for yy in (bot + (top - bot) / 3, bot + 2 * (top - bot) / 3):
        t = np.linspace(np.pi, 2 * np.pi, 40)
        ax.plot(cx + w / 2 * np.cos(t), yy + e / 2 * np.sin(t), color=INK, lw=1.2, zorder=3)
    ax.add_patch(Ellipse((cx, top), w, e, fc='white', ec=INK, lw=LW, zorder=3))


def model(cx, cy, r=5.4):
    """Model icon: a compact network in a circle."""
    ax.add_patch(Circle((cx, cy), r, fc='white', ec=INK, lw=LW + 0.4, zorder=3))
    left = [(cx - 2.9, cy + 1.5), (cx - 2.9, cy - 1.5)]
    mid = [(cx, cy + 2.7), (cx, cy), (cx, cy - 2.7)]
    right = [(cx + 2.9, cy)]
    for a, b in ((left, mid), (mid, right)):
        for p in a:
            for q in b:
                ax.plot([p[0], q[0]], [p[1], q[1]], color=INK, lw=1.3, zorder=3)
    for (x, y) in left + mid + right:
        ax.add_patch(Circle((x, y), 0.7, fc='white', ec=INK, lw=1.7, zorder=4))


def roc(x, y, s=10):
    ax.add_patch(Rectangle((x, y), s, s, fc='white', ec=INK, lw=LW - 0.6, zorder=2))
    f = np.linspace(0, 1, 80); curve = 1 - (1 - f) ** 3.2
    ax.fill_between(x + s * f, y + s * f, y + s * curve, color=FILL, zorder=2)
    ax.plot(x + s * f, y + s * curve, color=INK, lw=1.6, zorder=3)
    ax.plot([x, x + s], [y, y + s], color='#7a7a7a', lw=1, ls=(0, (2, 2)), zorder=3)
    text(x + s - 0.5, y + 1.3, 'ROC', size=13, ha='right')


def bracket(xs, y_top, y_bar, x_mid, y_end):
    for x in xs:
        ax.plot([x, x], [y_top, y_bar], color=INK, lw=LW, zorder=3, solid_capstyle='butt')
    ax.plot([min(xs), max(xs)], [y_bar, y_bar], color=INK, lw=LW, zorder=3, solid_capstyle='projecting')
    arrow(x_mid, y_bar, x_mid, y_end)


def slide_with_roi(x, y, w, h):
    """Slide thumbnail with a hand-drawn ROI and the tile grid inside it."""
    framed(thumb, x, y, w, h, radius=1.6, pad=0.9)
    ix0, iy0, iw, ih = x + 0.9, y + 0.9, w - 1.8, h - 1.8
    # ROI over the mucosa-rich upper-left lobe, in image-fraction coordinates
    roi = np.array([(0.08, 0.28), (0.18, 0.62), (0.34, 0.80), (0.52, 0.86), (0.66, 0.72),
                    (0.66, 0.46), (0.52, 0.28), (0.30, 0.18)])
    pts = np.c_[ix0 + roi[:, 0] * iw, iy0 + roi[:, 1] * ih]
    poly = Polygon(pts, closed=True, fill=False, ec=RED, lw=3.4, zorder=5, joinstyle='round')
    ax.add_patch(poly)
    clip = Polygon(pts, closed=True, transform=ax.transData, fc='none', ec='none')
    ax.add_patch(clip)
    step = 1.35
    for gx in np.arange(pts[:, 0].min(), pts[:, 0].max() + step, step):
        l, = ax.plot([gx, gx], [pts[:, 1].min(), pts[:, 1].max()], color=INK, lw=0.7, zorder=4)
        l.set_clip_path(clip)
    for gy in np.arange(pts[:, 1].min(), pts[:, 1].max() + step, step):
        l, = ax.plot([pts[:, 0].min(), pts[:, 0].max()], [gy, gy], color=INK, lw=0.7, zorder=4)
        l.set_clip_path(clip)


def tiles(x, y, cols, rows, size=4.3, gap=0.9, seed=5):
    """Tiles sampled from inside the ROI drawn on the slide thumbnail."""
    rng = np.random.default_rng(seed); h, w, _ = plain.shape; k = 44; n = 0
    while n < cols * rows:
        yy = rng.integers(int(0.16 * h), int(0.70 * h) - k)
        xx = rng.integers(int(0.10 * w), int(0.64 * w) - k)
        c = plain[yy:yy + k, xx:xx + k]
        g = c.mean(axis=2)
        if (g < 195).mean() > 0.92 and (g < 80).mean() < 0.02:   # tissue, no pigment/ink
            r, q = divmod(n, cols)
            x0, y0 = x + q * (size + gap), y + (rows - 1 - r) * (size + gap)
            ax.imshow(c, extent=(x0, x0 + size, y0, y0 + size), aspect='auto', zorder=2)
            ax.add_patch(Rectangle((x0, y0), size, size, fill=False, ec=INK, lw=1.4, zorder=3))
            n += 1


# ------------------------------------------------------------------ phase 1
slide_with_roi(9, 52, 26, 19)
arrow(22, 51.2, 22, 40.8)
text(20.2, 45.5, 'Tile extraction', ha='right')
text(23.8, 45.6, '+ Tissue detection\n+ Blur filter\n+ Stain normalization', size=13.5, linespacing=1.35)
tiles(9.5, 21, 5, 3)
arrow(22, 19.6, 22, 13.6)
text(20.2, 16.6, 'Features', ha='right')
store(22, 8.4)
text(22, 2.4, 'Feature bags', size=14.5, weight='bold', ha='center')

# ------------------------------------------------------------------ phase 2
for x in (67, 78, 89):
    store(x, 67.5)
arrow(78, 61.8, 78, 50.6)
text(76.2, 56.3, 'Training', ha='right')
text(79.8, 56.3, 'Classification,\nregression,\nor survival', size=13.5, linespacing=1.35)
model(78, 42.5)
text(78, 34.6, 'Model', size=14.5, weight='bold', ha='center')
for x in (93, 103):
    store(x, 42.5)
bracket([78, 98], 32.8, 29.5, 88, 21.2)
text(86.2, 25.3, 'Validation', ha='right')
roc(83, 8.2)
text(88, 2.4, 'Metrics', size=14.5, weight='bold', ha='center')

# ------------------------------------------------------------------ phase 3
model(125, 67)
text(125, 59.4, 'Model', size=14.5, weight='bold', ha='center')
for x in (141, 151):
    store(x, 67)
bracket([125, 146], 57.8, 54.8, 135.5, 48.2)
framed(att, 122, 30, 12.5, 18, radius=1.0, pad=0.5)
text(137.5, 39, 'Heatmaps')
framed(mosaic, 122, 15.5, 12.5, 11, radius=1.0, pad=0.5)
text(137.5, 21, 'Mosaic maps')
framed(umap, 122, 4, 12.5, 7.5, radius=1.0, pad=0.5)
text(137.5, 7.75, 'Feature-space\nanalysis', linespacing=1.25)

# ------------------------------------------------------------------ phases
for x, n in ((2.5, '1'), (60, '2'), (114, '3')):
    text(x, 3.4, n, size=32, weight='bold')
arrow(50, 42.5, 57, 42.5, color=ARROW_GREY, lw=3.4, ms=24)
arrow(108.5, 42.5, 115.5, 42.5, color=ARROW_GREY, lw=3.4, ms=24)

fig.savefig(OUT, dpi=150, facecolor='white', bbox_inches='tight', pad_inches=0.15)
print('wrote', OUT)
