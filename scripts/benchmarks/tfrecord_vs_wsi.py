"""Benchmark: extracting features directly from WSIs vs. from HistoX TFRecords.

Measures, under cold and warm OS page cache: one-time TFRecord extraction,
one feature-extraction pass reading tiles directly from the slides (with
parallel tile-reading processes), and one pass reading from TFRecords.

Usage:
    python scripts/benchmarks/tfrecord_vs_wsi.py SLIDE_DIR WORK_DIR [--slides 6]

SLIDE_DIR holds open-access TCGA diagnostic slides (*-DX1.*.svs). WORK_DIR is
a scratch directory (a temporary project with TFRecords is created there).
Writes WORK_DIR/results.json.
"""
import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch

import histox as hx

p = argparse.ArgumentParser()
p.add_argument('slide_dir', type=Path)
p.add_argument('work', type=Path)
p.add_argument('--slides', type=int, default=6)
p.add_argument('--tile-px', type=int, default=224)
p.add_argument('--tile-um', type=int, default=112)
p.add_argument('--batch-size', type=int, default=64)
p.add_argument('--repeats', type=int, default=1)
p.add_argument('--processes', type=int, default=16, help='tile-reading processes for the direct path')
args = p.parse_args()
PX, UM, BS, NPROC = args.tile_px, args.tile_um, args.batch_size, args.processes


def pick_slides():
    chosen, sites = [], set()
    for path in sorted(args.slide_dir.glob('TCGA-*-01Z-00-DX1.*.svs')):
        site = path.name.split('-')[1]
        if site in sites:
            continue
        try:
            hx.WSI(str(path), tile_px=PX, tile_um=UM, verbose=False)
        except Exception:
            continue
        chosen.append(path); sites.add(site)
        if len(chosen) == args.slides:
            return chosen
    raise RuntimeError('not enough slides')


def evict(paths):
    """Drop files from the OS page cache (no root needed) to measure cold reads."""
    for path in paths:
        path = Path(path)
        for f in ([path] if path.is_file() else [q for q in path.rglob('*') if q.is_file()]):
            fd = os.open(os.path.realpath(f), os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn):
    sync(); t = time.perf_counter(); out = fn(); sync()
    return time.perf_counter() - t, out


def wsi(path):
    w = hx.WSI(str(path), tile_px=PX, tile_um=UM, verbose=False)
    w.qc('otsu')
    return w


def wsi_features(slides, extractor):
    n = 0
    for s in slides:
        f = extractor(wsi(s), batch_size=BS, num_processes=NPROC)
        n += int((~np.ma.getmaskarray(f)[..., 0]).sum()) if np.ma.isMaskedArray(f) else int(np.prod(f.shape[:-1]))
    return n


def tfr_features(dataset, extractor, outdir):
    shutil.rmtree(outdir, ignore_errors=True)
    dataset.generate_feature_bags(extractor, str(outdir), batch_size=BS, force_regenerate=True)
    return sum(torch.load(f, map_location='cpu').shape[0] for f in Path(outdir).glob('*.pt'))


def dir_size(path):
    return sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())


def main():
    work = args.work; shutil.rmtree(work, ignore_errors=True); (work / 'slides').mkdir(parents=True)
    slides = pick_slides()
    for s in slides:
        os.symlink(s, work / 'slides' / s.name)
    ann = work / 'annotations.csv'
    ann.write_text('patient,slide,label\n' + ''.join(f'{s.name[:12]},{s.stem},a\n' for s in slides))
    project = hx.create_project(str(work / 'project'), annotations=str(ann), slides=str(work / 'slides'))
    extractor = hx.build_feature_extractor('resnet50_imagenet', tile_px=PX)
    res = {'histox_version': hx.__version__, 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           'slides': [s.name for s in slides], 'tile_px': PX, 'tile_um': UM, 'batch_size': BS,
           'extractor': 'resnet50_imagenet', 'qc': 'otsu', 'runs': []}

    res['processes_direct'] = NPROC
    tfr_dir = Path(project.root) / 'tfrecords'
    for cache in ('cold', 'warm'):
        for r in range(args.repeats):
            run = {'cache': cache}
            def prep(paths):
                if cache == 'cold':
                    evict(paths)
                else:   # warm: read once so the files are cached
                    for path in paths:
                        for f in ([Path(path)] if Path(path).is_file() else Path(path).rglob('*')):
                            if f.is_file():
                                Path(os.path.realpath(f)).read_bytes()
            shutil.rmtree(tfr_dir, ignore_errors=True)
            prep(slides)
            run['extract_s'], _ = timed(lambda: project.extract_tiles(PX, UM, qc='otsu', skip_extracted=False))
            dataset = project.dataset(PX, UM)
            run['tiles'] = int(dataset.num_tiles)
            run['tfrecord_bytes'] = dir_size(tfr_dir)
            prep(slides)
            run['wsi_features_s'], run['wsi_features_tiles'] = timed(lambda: wsi_features(slides, extractor))
            prep([tfr_dir])
            run['tfr_features_s'], run['tfr_features_tiles'] = timed(lambda: tfr_features(dataset, extractor, work / 'bags'))
            run = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in run.items()}
            res['runs'].append(run); print('run', run, flush=True)
            (work / 'results.json').write_text(json.dumps(res, indent=2))
    res['slide_bytes'] = sum(os.path.getsize(os.path.realpath(s)) for s in slides)
    (work / 'results.json').write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
