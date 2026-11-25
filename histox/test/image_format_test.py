import io
import unittest

import numpy as np
from PIL import Image

from histox.io.io_utils import detect_image_format


def _encode(fmt):
    buf = io.BytesIO()
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(buf, format=fmt)
    return buf.getvalue()


class DetectImageFormatTest(unittest.TestCase):

    def test_formats(self):
        self.assertEqual(detect_image_format(_encode('JPEG')), 'jpeg')
        self.assertEqual(detect_image_format(_encode('PNG')), 'png')
        self.assertEqual(detect_image_format(_encode('TIFF')), 'tiff')
        self.assertEqual(detect_image_format(_encode('BMP')), 'bmp')

    def test_unknown(self):
        self.assertIsNone(detect_image_format(b'not an image'))
        self.assertIsNone(detect_image_format(b''))


if __name__ == '__main__':
    unittest.main()
