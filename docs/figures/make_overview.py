"""Draw the HistoX workflow overview (docs/source/overview.svg).

Hand-built SVG so strokes, corners, arrowheads and type stay crisp at any
size. Embedded images are HistoX outputs generated from open-access TCGA
slides (see the generation metadata in docs/source).

Run from the repository root:  python docs/figures/make_overview.py
"""
import base64
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
SRC = ROOT / 'docs/source'
OUT = SRC / 'overview.svg'

W, H = 1756, 800
INK, GREY, FILL, RED = '#000', '#a8a8a8', '#e6e6e6', '#b4141c'
SW = 4            # main stroke width
FONT = "'Helvetica Neue', Helvetica, Arial, 'Liberation Sans', sans-serif"
out = []


def emit(s):
    out.append(s)


def data_uri(im, fmt='JPEG', quality=88):
    buf = io.BytesIO()
    im.save(buf, format=fmt, quality=quality) if fmt == 'JPEG' else im.save(buf, format=fmt, optimize=True)
    return f'data:image/{fmt.lower()};base64,' + base64.b64encode(buf.getvalue()).decode()


def text(x, y, s, size=31, weight=400, anchor='start'):
    emit(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}">{s}</text>')


def arrow(x0, y0, x1, y1, color=INK, width=SW, head=16):
    """Straight arrow with a solid triangular head ending exactly at (x1, y1)."""
    d = np.array([x1 - x0, y1 - y0], float); L = np.hypot(*d); u = d / L; n = np.array([-u[1], u[0]])
    base = np.array([x1, y1]) - u * head * 1.25
    emit(f'<line x1="{x0}" y1="{y0}" x2="{base[0]:.1f}" y2="{base[1]:.1f}" stroke="{color}" stroke-width="{width}"/>')
    p = [np.array([x1, y1]), base + n * head * 0.62, base - n * head * 0.62]
    emit('<polygon points="' + ' '.join(f'{q[0]:.1f},{q[1]:.1f}' for q in p) + f'" fill="{color}"/>')


def bracket(xs, y0, ybar, xmid, yend):
    pts = ' '.join(f'{x},{y}' for x, y in [(xs[0], y0), (xs[0], ybar), (xs[1], ybar), (xs[1], y0)])
    emit(f'<polyline points="{pts}" fill="none" stroke="{INK}" stroke-width="{SW}" stroke-linejoin="miter"/>')
    arrow(xmid, ybar, xmid, yend)


def framed_image(uri, x, y, w, h, r=10, inset=6, bg='#ffffff', aspect='xMidYMid slice'):
    cid = f'clip{len(out)}'
    emit(f'<clipPath id="{cid}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}"/></clipPath>')
    emit(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{bg}"/>')
    emit(f'<image href="{uri}" x="{x + inset}" y="{y + inset}" width="{w - 2 * inset}" height="{h - 2 * inset}" '
         f'preserveAspectRatio="{aspect}" clip-path="url(#{cid})"/>')
    emit(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="none" stroke="{INK}" stroke-width="{SW}"/>')


def stack(cx, cy, s=86):
    """Data icon: three offset tiles with a 3x3 grid on the front tile."""
    off = s * 0.13
    for k in (2, 1):
        x, y = cx - s / 2 + k * off, cy - s / 2 - k * off
        emit(f'<rect x="{x:.1f}" y="{y:.1f}" width="{s}" height="{s}" rx="7" fill="#f4f4f4" stroke="{INK}" stroke-width="3"/>')
    x, y = cx - s / 2, cy - s / 2
    emit(f'<rect x="{x:.1f}" y="{y:.1f}" width="{s}" height="{s}" rx="7" fill="{FILL}" stroke="{INK}" stroke-width="{SW}"/>')
    for i in (1, 2):
        emit(f'<line x1="{x + i * s / 3:.1f}" y1="{y + 3}" x2="{x + i * s / 3:.1f}" y2="{y + s - 3}" stroke="{INK}" stroke-width="2"/>')
        emit(f'<line x1="{x + 3}" y1="{y + i * s / 3:.1f}" x2="{x + s - 3}" y2="{y + i * s / 3:.1f}" stroke="{INK}" stroke-width="2"/>')


def model(cx, cy, r=58):
    """Model icon: three inputs merging into one output, inside a circle."""
    emit(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#fff" stroke="{INK}" stroke-width="{SW + 0.5}"/>')
    nx, ox, ny = cx - 22, cx + 22, [cy - 22, cy, cy + 22]
    for y in ny:
        emit(f'<path d="M{nx + 8},{y} C{cx},{y} {cx - 4},{cy} {ox - 9},{cy}" fill="none" stroke="{INK}" stroke-width="3"/>')
    for y in ny:
        emit(f'<rect x="{nx - 9}" y="{y - 8}" width="18" height="16" rx="5" fill="#fff" stroke="{INK}" stroke-width="3"/>')
    emit(f'<rect x="{ox - 10}" y="{cy - 10}" width="22" height="20" rx="6" fill="{INK}"/>')


def roc(x, y, s=104):
    emit(f'<rect x="{x}" y="{y}" width="{s}" height="{s}" fill="#fff" stroke="{INK}" stroke-width="2.5"/>')
    f = np.linspace(0, 1, 60); c = 1 - (1 - f) ** 3.2
    curve = ' '.join(f'{x + s * a:.1f},{y + s - s * b:.1f}' for a, b in zip(f, c))
    diag = f'{x + s:.1f},{y:.1f} {x:.1f},{y + s:.1f}'
    emit(f'<polygon points="{curve} {diag}" fill="{FILL}"/>')
    emit(f'<polyline points="{curve}" fill="none" stroke="{INK}" stroke-width="2.5"/>')
    emit(f'<line x1="{x}" y1="{y + s}" x2="{x + s}" y2="{y}" stroke="#777" stroke-width="1.5" stroke-dasharray="4 4"/>')
    text(x + s - 6, y + s - 9, 'ROC', size=21, anchor='end')


def scatter_icon(x, y, w, h):
    emit(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{FILL}" stroke="{INK}" stroke-width="{SW}"/>')
    rng = np.random.default_rng(3)
    for (cx, cy, col) in ((x + w * 0.30, y + h * 0.62, '#e4572e'), (x + w * 0.62, y + h * 0.36, '#3a6ea5'),
                          (x + w * 0.72, y + h * 0.70, '#7b2d74')):
        for dx, dy in rng.normal(0, 1, (14, 2)):
            emit(f'<circle cx="{cx + dx * w * 0.07:.1f}" cy="{cy + dy * h * 0.08:.1f}" r="4.2" fill="{col}"/>')


def smooth_closed(pts):
    """Catmull-Rom spline through pts as a closed cubic Bezier path."""
    p = list(pts); n = len(p); d = f'M{p[0][0]:.1f},{p[0][1]:.1f}'
    for i in range(n):
        p0, p1, p2, p3 = p[i - 1], p[i], p[(i + 1) % n], p[(i + 2) % n]
        c1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
        d += f' C{c1[0]:.1f},{c1[1]:.1f} {c2[0]:.1f},{c2[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}'
    return d + ' Z'


# ------------------------------------------------------------------ images
full = Image.open(SRC / '_static/slide_processing/tutorial_thumbnail.png').convert('RGB')
plain = np.asarray(full)
thumb = full.crop((0, 110, 800, 560))          # trim empty glass above/below the tissue
TW, TH = thumb.size
att = np.asarray(Image.open(SRC / '_static/examples/ps3_spatial_attention_tcga_a6_5660.png')
                 .convert('RGB').crop((55, 40, 385, 525))).copy()
hsv = np.asarray(Image.fromarray(att).convert('HSV'))
att[hsv[..., 1] < 40] = (230, 230, 230)
att_img = Image.fromarray(att).rotate(90, expand=True, fillcolor=(230, 230, 230))
mosaic = Image.open(SRC / 'mosaic_example.png').convert('RGB').crop((400, 770, 900, 1140))   # densest part
mosaic = Image.fromarray(np.where(np.asarray(mosaic).mean(axis=2, keepdims=True) > 247,
                                  np.uint8(230), np.asarray(mosaic)).astype(np.uint8))

emit(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
     f'font-family="{FONT}" fill="{INK}">')
emit(f'<rect width="{W}" height="{H}" fill="#fff"/>')

# ------------------------------------------------------------------ phase 1
sx, sy, sw, sh = 116, 47, 312, 177
framed_image(data_uri(thumb), sx, sy, sw, sh, r=16, inset=8, aspect='none')
ix, iy, iw, ih = sx + 8, sy + 8, sw - 16, sh - 16
# ROI over the mucosa and tumour in the left half, in cropped-thumbnail pixels
roi_px = [(30, 330), (60, 170), (170, 120), (300, 110), (420, 170), (440, 280), (380, 380), (230, 410), (90, 400)]
roi = [(ix + x / TW * iw, iy + y / TH * ih) for x, y in roi_px]
path = smooth_closed(roi)
emit(f'<clipPath id="roi"><path d="{path}"/></clipPath>')
xs, ys = [p[0] for p in roi], [p[1] for p in roi]
g = '<g clip-path="url(#roi)" stroke="#111" stroke-width="1.3">'
for gx in np.arange(min(xs) - 2, max(xs) + 12, 11.5):
    g += f'<line x1="{gx:.1f}" y1="{min(ys) - 5:.1f}" x2="{gx:.1f}" y2="{max(ys) + 5:.1f}"/>'
for gy in np.arange(min(ys) - 2, max(ys) + 12, 11.5):
    g += f'<line x1="{min(xs) - 5:.1f}" y1="{gy:.1f}" x2="{max(xs) + 5:.1f}" y2="{gy:.1f}"/>'
emit(g + '</g>')
emit(f'<path d="{path}" fill="none" stroke="{RED}" stroke-width="7" stroke-linejoin="round"/>')

arrow(271, 224, 271, 330)
text(248, 294, 'Tile extraction', anchor='end')
for i, s in enumerate(('+ Tissue detection', '+ Blur filter', '+ Stain normalization')):
    text(300, 262 + i * 30, s, size=23)

rng = np.random.default_rng(11); k = 40; tiles = []
while len(tiles) < 24:
    yy = int(rng.integers(int(0.30 * TH), int(0.68 * TH) - k)); xx = int(rng.integers(int(0.08 * TW), int(0.45 * TW) - k))
    c = plain[yy:yy + k, xx:xx + k]; gch = c.mean(axis=2)
    if (gch < 200).mean() > 0.9 and (gch < 70).mean() < 0.02:
        tiles.append(Image.fromarray(c).resize((72, 72), Image.LANCZOS))
for i, t in enumerate(tiles):
    r, q = divmod(i, 6); x, y = 147 + q * 43, 347 + r * 46
    emit(f'<image href="{data_uri(t, quality=85)}" x="{x}" y="{y}" width="36" height="36" preserveAspectRatio="none"/>')
    emit(f'<rect x="{x}" y="{y}" width="36" height="36" fill="none" stroke="{INK}" stroke-width="2"/>')

arrow(271, 548, 271, 603)
text(248, 580, 'Features', anchor='end')
stack(266, 672)
text(271, 752, 'Feature bags', size=24, weight=700, anchor='middle')
text(48, 756, '1', size=46, weight=700)

# ------------------------------------------------------------------ phase 2
for cx in (637, 762, 887):
    stack(cx - 5, 145)
arrow(762, 222, 762, 300)
text(738, 270, 'Training', anchor='end')
for i, s in enumerate(('Classification,', 'regression,', 'or survival')):
    text(791, 240 + i * 30, s, size=23)
model(762, 387)
text(762, 470, 'Model', size=24, weight=700, anchor='middle')
for cx in (927, 1052):
    stack(cx - 5, 392)
bracket((762, 1001), 495, 515, 883, 600)
text(850, 568, 'Validation', anchor='end')
roc(830, 620)
text(882, 752, 'Metrics', size=24, weight=700, anchor='middle')
text(586, 756, '2', size=46, weight=700)

# ------------------------------------------------------------------ phase 3
model(1246, 143)
text(1246, 228, 'Model', size=24, weight=700, anchor='middle')
for cx in (1456, 1582):
    stack(cx - 5, 148)
bracket((1246, 1521), 242, 262, 1385, 340)
framed_image(data_uri(att_img), 1306, 360, 162, 100, r=10, inset=4, bg='#e6e6e6', aspect='xMidYMid meet')
text(1488, 421, 'Heatmaps')
framed_image(data_uri(mosaic), 1312, 489, 150, 112, r=10, inset=4, bg='#e6e6e6', aspect='xMidYMid slice')
text(1488, 556, 'Mosaic maps')
scatter_icon(1311, 642, 152, 98)
text(1488, 684, 'Feature-space'); text(1488, 722, 'analysis')
text(1174, 756, '3', size=46, weight=700)

# ------------------------------------------------------------------ between phases
arrow(495, 405, 585, 405, color=GREY, width=6, head=20)
arrow(1158, 405, 1248 - 8, 405, color=GREY, width=6, head=20)

emit('</svg>')
OUT.write_text('\n'.join(out))
print('wrote', OUT, f'{OUT.stat().st_size / 1024:.0f} KB')
