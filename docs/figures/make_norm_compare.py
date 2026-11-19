"""Generate the stain-normalization example figures (docs/source/norm_compare/).

Uses HistoX's own WSI reader and StainNormalizer on open-access TCGA
diagnostic slides from nine different tissue source sites. Writes the
figures plus generation_metadata.json describing exactly what was used.

Usage (from the repository root):
    python docs/figures/make_norm_compare.py /path/to/open/TCGA/slides
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

import histox as hx

SLIDE_DIR = Path(sys.argv[1])
OUT = Path('docs/source/norm_compare')
SITES = ['AA', 'AG', 'A6', 'CM', 'D5', 'G4', 'DM', 'AZ', 'AF']   # TCGA tissue source sites
GRID_PX = 1200
METHODS = {   # file stem: (normalizer, label)
    'reinhard': ('reinhard', 'Reinhard'),
    'macenko': ('macenko', 'Macenko'),
}
# Vahadane (sklearn) is omitted until its stain-matrix estimation is fixed: it
# currently applies sparsity to the stain vectors rather than the
# concentrations and produces colour casts on some tiles.
COMPARE = [('macenko', 'Macenko'), ('macenko_fast', 'Macenko Fast'),
           ('reinhard', 'Reinhard'), ('reinhard_mask', 'Reinhard Mask'),
           ('reinhard_fast', 'Reinhard Fast'), ('reinhard_fast_mask', 'Reinhard Fast Mask')]


def pick_slides():
    chosen = []
    for site in SITES:
        for hit in sorted(SLIDE_DIR.glob(f'TCGA-{site}-*-01Z-00-DX1.*.svs')):
            try:   # some early TCGA scans lack MPP metadata
                hx.WSI(str(hit), tile_px=256, tile_um=128, verbose=False)
            except Exception:
                continue
            chosen.append(hit)
            break
    assert len(chosen) == 9, chosen
    return chosen


def thumbnail(path, width=480):
    wsi = hx.WSI(str(path), tile_px=256, tile_um=128, verbose=False)
    im = wsi.thumb(width=width).convert('RGB')
    return np.asarray(square(im, (255, 255, 255)))


def square(im, fill):
    s = max(im.size)
    out = Image.new('RGB', (s, s), fill)
    out.paste(im, ((s - im.width) // 2, (s - im.height) // 2))
    return out


def tissue_tile(path, px=480, um=240):
    """A tissue-rich tile chosen deterministically after Otsu QC."""
    wsi = hx.WSI(str(path), tile_px=px, tile_um=um, verbose=False)
    wsi.qc('otsu')
    best, best_score = None, -1
    for i, t in enumerate(wsi.build_generator(shuffle=False, img_format='numpy')()):
        img = t['image']
        gray = img.mean(axis=2)
        tissue = ((gray < 215) & (gray > 40)).mean()
        score = tissue - abs(gray.mean() - 165) / 400
        if score > best_score:
            best, best_score = img, score
        if i >= 400:
            break
    return np.asarray(best)


def normalize(img, method, preset='v1'):
    n = hx.norm.autoselect(method, source=preset)
    return n.rgb_to_rgb(img)


def grid(images, path):
    k = GRID_PX // 3
    canvas = Image.new('RGB', (GRID_PX, GRID_PX), 'white')
    for i, img in enumerate(images):
        r, c = divmod(i, 3)
        canvas.paste(Image.fromarray(img).resize((k - 8, k - 8), Image.LANCZOS), (c * k + 4, r * k + 4))
    canvas.save(path, quality=85, optimize=True)


def compare(img, path):
    panels = [('Original', img)] + [(lab, normalize(img, m)) for m, lab in COMPARE]
    fig, axes = plt.subplots(2, 4, figsize=(14.4, 7.6), dpi=100)
    for ax in axes.flat:
        ax.axis('off')
    for ax, (lab, im) in zip(axes.flat, panels):
        ax.imshow(im); ax.set_title(lab, fontsize=13, color='#262a2e')
    fig.tight_layout()
    fig.savefig(path, pil_kwargs={'quality': 85, 'optimize': True}, facecolor='white')
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    slides = pick_slides()
    thumbs = [thumbnail(p) for p in slides]
    tiles = [tissue_tile(p) for p in slides]
    written = []
    grid(thumbs, OUT / 'wsi_unnormalized.jpg'); written.append('wsi_unnormalized.jpg')
    grid(tiles, OUT / 'tile_unnormalized.jpg'); written.append('tile_unnormalized.jpg')
    for stem, (method, _) in METHODS.items():
        for preset in ('v1', 'v2'):
            grid([normalize(t, method, preset) for t in thumbs], OUT / f'wsi_{stem}_{preset}.jpg')
            tile_method = 'reinhard_mask' if stem == 'reinhard' else method
            grid([normalize(t, tile_method, preset) for t in tiles], OUT / f'tile_{stem}_{preset}.jpg')
            written += [f'wsi_{stem}_{preset}.jpg', f'tile_{stem}_{preset}.jpg']
    compare(thumbs[0], OUT / 'wsi_norm_compare.jpg')
    compare(tiles[0], OUT / 'tile_norm_compare.jpg')
    written += ['wsi_norm_compare.jpg', 'tile_norm_compare.jpg']
    meta = {
        'description': 'Stain-normalization examples generated with HistoX.',
        'histox_version': hx.__version__,
        'script': 'docs/figures/make_norm_compare.py',
        'data': {
            'program': 'TCGA', 'data_portal': 'NCI Genomic Data Commons', 'access': 'open',
            'slides': [{'filename': p.name, 'tissue_source_site': p.name.split('-')[1]} for p in slides],
        },
        'thumbnail_width_px': 480,
        'tile': {'tile_px': 480, 'tile_um': 240, 'qc': 'otsu', 'selection': 'highest tissue score in first 400 tiles'},
        'grid_order': 'row-major, same slide order as data.slides',
        'comparison_figures_use': 'first slide in data.slides',
        'presets': ['v1', 'v2'],
        'tile_reinhard_uses': 'reinhard_mask',
        'files': {f: hashlib.sha256((OUT / f).read_bytes()).hexdigest() for f in sorted(written)},
    }
    (OUT / 'generation_metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    print('wrote', len(written), 'figures')


if __name__ == '__main__':
    main()
