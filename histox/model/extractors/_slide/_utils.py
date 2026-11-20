"""Utility functions for slide feature extraction."""

import numpy as np
import histox as hx
from histox import log

# -----------------------------------------------------------------------------

def _build_grid(extractor, slide, grid=None, dtype=np.float16):
    """Build a grid of features for a slide."""

    total_out = (extractor.num_features
                 + extractor.num_classes
                 + extractor.num_uncertainty)
    if grid is None:
        features_grid = np.ma.masked_all((
                slide.grid.shape[1],
                slide.grid.shape[0],
                total_out),
            dtype=dtype)
    else:
        assert grid.shape == (slide.grid.shape[1], slide.grid.shape[0], total_out)
        features_grid = grid
    return features_grid


def _log_normalizer(normalizer):
    """Log the stain normalizer being used."""
    if normalizer is None:
        log.debug("Calculating slide features without stain normalization")
    else:
        log.debug(
            f"Calculating slide features using stain normalizer: {normalizer}"
        )

def _use_numpy_if_png(img_format):
    """Use numpy image format instead of PNG."""
    if img_format == 'png':  # PNG is lossless; this is equivalent but faster
        log.debug("Using numpy image format instead of PNG")
        return 'numpy'
    return img_format


def _tfrecord_img_format(img_format):
    """Image format to read tiles in when they are also written to a TFRecord.

    TFRecords store encoded image bytes, so tiles are read as JPEG (the same
    format :meth:`histox.Dataset.extract_tiles` uses by default) and the model
    sees exactly the tiles that are stored.
    """
    if img_format in (None, 'numpy') or img_format.lower() in ('jpg', 'jpeg'):
        if img_format == 'numpy':
            log.info("Writing a TFRecord: reading tiles as JPEG so the model "
                     "sees the same tiles that are stored.")
        return 'jpg'
    raise ValueError(
        "tfrecord_dir supports img_format='jpg'. For lossless PNG TFRecords, "
        "use Dataset.extract_tiles(img_format='png')."
    )


def _tee_to_tfrecord(generator, slide, tfrecord_dir):
    """Wrap a tile generator so every tile it yields is also written to a TFRecord.

    Records use the same layout as :meth:`histox.WSI.extract_tiles`
    (``image_raw``, ``slide``, ``loc_x``, ``loc_y``). The file is written to a
    temporary ``.partial`` path and renamed to ``<slide>.tfrecords`` only when
    the generator is exhausted, so an interrupted run never leaves a truncated
    TFRecord behind. An index is created after the rename.
    """
    import os
    from os.path import join
    from histox.util import tfrecord2idx

    os.makedirs(tfrecord_dir, exist_ok=True)
    path = join(tfrecord_dir, slide.name + '.tfrecords')
    tmp = path + '.partial'
    slide_bytes = bytes(slide.name, 'utf-8')

    def _generator():
        writer = hx.io.TFRecordWriter(tmp)
        n, complete = 0, False
        try:
            for image_dict in generator():
                x, y = image_dict['loc']
                writer.write(hx.io.serialized_record(
                    slide_bytes, image_dict['image'], x, y))
                n += 1
                yield image_dict
            complete = True
        finally:
            writer.close()
            if complete and n:
                os.replace(tmp, path)
                stale = tfrecord2idx.find_index(path)
                if stale:
                    os.remove(stale)
                tfrecord2idx.create_index(path, join(tfrecord_dir, slide.name + '.index'))
                log.info(f"Wrote {n} tiles to [green]{path}")
            else:
                if os.path.exists(tmp):
                    os.remove(tmp)
                if complete:
                    log.info(f"No tiles extracted for [green]{slide.name}")
    return _generator
