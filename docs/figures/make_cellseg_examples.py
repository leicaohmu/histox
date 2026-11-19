"""Generate the masked / unmasked cell examples (docs/source/cell_*.png).

1. Find the most nucleus-dense 256 um region (40x) of the open-access TCGA
   slide used by the slide-processing guide (``tcga-coad-slide-processing``
   in histox.data).
2. Save that region as a small pyramidal TIFF and segment it with HistoX's
   Cellpose integration.
3. Extract cell images with and without masks and keep one representative
   nucleus from both runs.

Usage (from the repository root):
    python docs/figures/make_cellseg_examples.py /path/to/TCGA-3L-AA1B-01Z-00-DX1.8923A151-A690-40B7-9E5A-FCBEDFC2394F.svs
"""
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyvips
from PIL import Image

import histox as hx
from histox.cellseg import segment_slide

OUT = Path('docs/source')
SLIDE = Path(sys.argv[1])
MPP = 0.25                       # 40x
REGION_PX = 1024                 # 256 um
CELL_PX, CELL_UM = 96, 24        # extracted cell images
WINDOW_PX, WINDOW_UM = 256, 102  # Cellpose window; cyto2 diam_mean 30 px -> ~12 um cells


def densest_region():
    wsi = hx.WSI(str(SLIDE), tile_px=REGION_PX, tile_um=int(REGION_PX * MPP), verbose=False)
    wsi.qc('otsu')
    best, best_score = None, -1
    for i, t in enumerate(wsi.build_generator(shuffle=False, img_format='numpy')()):
        a = t['image'].astype(float)
        gray = a.mean(axis=2)
        nuclear = ((gray < 130) & (a[..., 2] > a[..., 1] + 10)).mean()
        background = (gray > 220).mean()
        score = nuclear - background
        if score > best_score:
            best, best_score, loc = t['image'], score, (t.get('loc_x'), t.get('loc_y'))
        if i >= 3000:
            break
    return np.ascontiguousarray(best), loc


def main():
    region, loc = densest_region()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        tiff = tmp / 'region.tiff'
        pyvips.Image.new_from_array(region).tiffsave(str(tiff), tile=True, pyramid=True,
                                                     compression='jpeg', Q=95)
        seg_wsi = hx.WSI(str(tiff), tile_px=WINDOW_PX, tile_um=WINDOW_UM, mpp=MPP, verbose=False)
        seg = segment_slide(seg_wsi, 'cyto2', gpus=(0,), spawn_workers=False, show_progress=False)
        wsi = hx.WSI(str(tiff), tile_px=CELL_PX, tile_um=CELL_UM, mpp=MPP, verbose=False)
        wsi.apply_segmentation(seg)
        for masked in (True, False):
            wsi.extract_cells(tiles_dir=str(tmp / ('masked' if masked else 'unmasked')),
                              img_format='png', report=False, apply_masks=masked)
        masked = sorted((tmp / 'masked').rglob('*.png'))
        unmasked = {p.name: p for p in (tmp / 'unmasked').rglob('*.png')}
        # Representative nucleus: compact, roughly elliptical, dark purple.
        candidates = []
        for p in masked:
            if p.name not in unmasked:
                continue
            a = np.asarray(Image.open(p).convert('RGB')).astype(float)
            inside = a.sum(axis=2) > 0
            frac = inside.mean()
            if not 0.08 <= frac <= 0.35:
                continue
            ys, xs = np.nonzero(inside)
            h, w = np.ptp(ys) + 1, np.ptp(xs) + 1
            fill, aspect = inside.sum() / (h * w), max(h, w) / min(h, w)
            r, g, b = (a[inside][:, i].mean() for i in range(3))
            if fill < 0.6 or aspect > 1.6 or not (r > g + 15 and b > g + 15):
                continue
            candidates.append((-(r + g + b) / 3 - 40 * (aspect - 1), p))
        candidates.sort(key=lambda c: c[0], reverse=True)
        best = candidates[0][1]
        sheet = Image.new('RGB', (12 * 100, 200), 'white')
        for i, (_, p) in enumerate(candidates[:12]):
            sheet.paste(Image.open(p).convert('RGB'), (i * 100, 0))
            sheet.paste(Image.open(unmasked[p.name]).convert('RGB'), (i * 100, 100))
        sheet.save('/tmp/hx_cell_candidates.png')
        Image.open(best).convert('RGB').save(OUT / 'cell_masked.png')
        Image.open(unmasked[best.name]).convert('RGB').save(OUT / 'cell_unmasked.png')
        n_cells = len(masked)
    meta = {
        'description': 'Cell image examples generated with HistoX cell segmentation.',
        'histox_version': hx.__version__,
        'script': 'docs/figures/make_cellseg_examples.py',
        'data': {'dataset': 'tcga-coad-slide-processing', 'file': SLIDE.name, 'program': 'TCGA',
                 'data_portal': 'NCI Genomic Data Commons', 'access': 'open',
                 'region': {'tile_px': REGION_PX, 'mpp': MPP, 'location': loc,
                            'selection': 'highest nuclear-pixel score in the first 3000 Otsu-retained tiles'}},
        'segmentation': {'model': 'cyto2', 'window_px': WINDOW_PX, 'window_um': WINDOW_UM},
        'extraction': {'tile_px': CELL_PX, 'tile_um': CELL_UM, 'cells_extracted': n_cells,
                       'candidates_after_filtering': len(candidates), 'selected_file': best.name},
        'files': {f: hashlib.sha256((OUT / f).read_bytes()).hexdigest() for f in ('cell_masked.png', 'cell_unmasked.png')},
    }
    (OUT / 'cellseg_generation_metadata.json').write_text(json.dumps(meta, indent=2, default=str) + '\n')
    print('selected', best.name, 'of', n_cells, 'cells;', len(candidates), 'candidates')


if __name__ == '__main__':
    main()
