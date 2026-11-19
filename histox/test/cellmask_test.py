import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import histox as hx


@unittest.skipUnless(hx.slide_backend() == 'libvips', 'requires the libvips slide backend')
class TestMaskedCellExtraction(unittest.TestCase):
    """extract_cells(apply_masks=True) keeps pixel values and blanks the rest."""

    def test_label_masks_do_not_rescale_pixels(self):
        import pyvips
        from histox.cellseg import Segmentation
        colour = (200, 100, 150)
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        image[:] = colour
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            tiff = tmp / 'slide.tiff'
            pyvips.Image.new_from_array(image).tiffsave(str(tiff), tile=True, pyramid=True)
            wsi = hx.WSI(str(tiff), tile_px=64, tile_um=32, mpp=0.5, verbose=False)
            masks = np.zeros((512, 512), dtype=np.int32)
            masks[236:276, 236:276] = 7          # one cell, label 7
            wsi.apply_segmentation(Segmentation(masks, wsi_dim=(512, 512), wsi_offset=(0, 0)))
            wsi.extract_cells(tiles_dir=str(tmp / 'cells'), img_format='png',
                              report=False, apply_masks=True)
            cells = list((tmp / 'cells').rglob('*.png'))
            self.assertEqual(len(cells), 1)
            cell = np.asarray(Image.open(cells[0]).convert('RGB'))
            inside = cell[32, 32]
            np.testing.assert_array_equal(inside, colour)
            np.testing.assert_array_equal(cell[0, 0], (0, 0, 0))


if __name__ == '__main__':
    unittest.main()
