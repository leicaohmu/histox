"""Generate the UMAP and mosaic-map examples (docs/source/umap_example.png,
docs/source/mosaic_example.png).

Tiles come from open-access TCGA diagnostic slides of three cancer types.
Features are ImageNet ResNet-50 activations from HistoX's feature-extractor
registry; HistoX's SlideMap and Mosaic draw the figures.

Usage (from the repository root):
    python docs/figures/make_feature_maps.py COLORECTAL_DIR LUNG_DIR PROSTATE_DIR
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import umap

import histox as hx

OUT = Path('docs/source')
SLIDES_PER_GROUP, TILES_PER_SLIDE, TILE_PX, TILE_UM = 6, 60, 224, 224
COLORECTAL_SITES = ('AA', 'AG', 'A6', 'CM', 'D5', 'G4', 'DM', 'AZ', 'AF')
GROUPS = [('Colorectal (TCGA-COAD/READ)', Path(sys.argv[1]), COLORECTAL_SITES),
          ('Lung (TCGA-LUAD/LUSC)', Path(sys.argv[2]), None),
          ('Prostate (TCGA-PRAD)', Path(sys.argv[3]), None)]
COLORS = ['#ee6c4d', '#4c78a8', '#7f2d73']


def open_slides(folder, sites):
    out = []
    for path in sorted(folder.glob('TCGA-*-01Z-00-DX1*.svs')):
        if sites and path.name.split('-')[1] not in sites:
            continue
        try:
            wsi = hx.WSI(str(path), tile_px=TILE_PX, tile_um=TILE_UM, verbose=False)
        except Exception:
            continue   # e.g. early scans without MPP metadata
        out.append((path, wsi))
        if len(out) == SLIDES_PER_GROUP:
            return out
    raise RuntimeError(f'not enough slides in {folder}')


def tiles_from(wsi):
    wsi.qc('otsu')
    gen = wsi.build_generator(shuffle=False, img_format='numpy', whitespace_fraction=0.6)()
    tiles = [t['image'] for i, t in enumerate(gen) if i < 1200]
    step = max(1, len(tiles) // TILES_PER_SLIDE)
    return tiles[::step][:TILES_PER_SLIDE]


def main():
    extractor = hx.build_feature_extractor('resnet50_imagenet', tile_px=TILE_PX)
    images, feats, slides, used = [], [], [], []
    labels = {}
    for (label, folder, sites) in GROUPS:
        for path, wsi in open_slides(folder, sites):
            tiles = tiles_from(wsi)
            batch = torch.from_numpy(np.stack(tiles)).permute(0, 3, 1, 2).contiguous()
            with torch.inference_mode():
                f = extractor(batch)
            f = (f[0] if isinstance(f, (list, tuple)) else f).float().cpu().numpy()
            images += tiles; feats.append(f)
            slides += [path.stem] * len(tiles)
            labels[path.stem] = label
            used.append({'filename': path.name, 'group': label, 'tiles': len(tiles)})
    feats = np.concatenate(feats)
    xy = umap.UMAP(n_neighbors=30, min_dist=0.2, random_state=42).fit_transform(feats)
    xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0))

    sm = hx.SlideMap.from_xy(xy[:, 0], xy[:, 1], slides=np.array(slides), tfr_index=np.arange(len(slides)))
    sm.label_by_slide(labels)
    fig, ax = plt.subplots(figsize=(9, 6.5), dpi=150)
    sm.plot(ax=ax, cmap={g[0]: c for g, c in zip(GROUPS, COLORS)}, s=12, alpha=0.9, linewidth=0,
            legend='Cancer type', loc='center right',
            legend_kwargs={'markerscale': 2.5, 'frameon': False})
    ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / 'umap_example.png', facecolor='white'); plt.close(fig)

    mosaic = hx.Mosaic(images, coords=xy, num_tiles_x=30)
    mosaic.save(str(OUT / 'mosaic_example.png'), figsize=(12, 12), background='#ffffff')

    meta = {
        'description': 'Feature-space examples generated with HistoX.',
        'histox_version': hx.__version__,
        'script': 'docs/figures/make_feature_maps.py',
        'data': {'program': 'TCGA', 'data_portal': 'NCI Genomic Data Commons', 'access': 'open', 'slides': used},
        'tiles': {'tile_px': TILE_PX, 'tile_um': TILE_UM, 'qc': 'otsu', 'per_slide': TILES_PER_SLIDE,
                  'selection': 'evenly spaced over the first 1200 retained tiles'},
        'features': 'resnet50_imagenet (histox.build_feature_extractor)',
        'umap': {'n_neighbors': 30, 'min_dist': 0.2, 'random_state': 42},
        'mosaic': {'num_tiles_x': 30, 'figsize_in': [12, 12]},
        'files': {f: hashlib.sha256((OUT / f).read_bytes()).hexdigest() for f in ('umap_example.png', 'mosaic_example.png')},
    }
    (OUT / 'feature_maps_generation_metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    print('tiles', len(images))


if __name__ == '__main__':
    main()
