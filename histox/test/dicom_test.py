import sys
import tempfile
import types
import unittest

from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

import histox as hx
from histox.slide.backends.dicom import _DicomReader


class _Size:
    def __init__(self, width, height):
        self.width = width
        self.height = height


class _Level:
    def __init__(self, level, width, height, mpp):
        self.level = level
        self.size = _Size(width, height)
        self.mpp = _Size(mpp, mpp)


class _FakeSlide:
    def __init__(self):
        self.size = _Size(1000, 800)
        self.mpp = _Size(0.25, 0.25)
        self.levels = [
            _Level(0, 1000, 800, 0.25),
            _Level(2, 250, 200, 1.0),
        ]
        self.closed = False

    def read_region(self, location, level, size, threads=1):
        del location, level, threads
        return Image.new("RGB", size, (40, 80, 120))

    def read_thumbnail(self, size):
        return Image.new("RGB", size, (20, 40, 60))

    def close(self):
        self.closed = True


class _FakeWsiDicom:
    slide = _FakeSlide()

    @classmethod
    def open(cls, path):
        del path
        return cls.slide


class TestDicomPaths(unittest.TestCase):
    def test_dicom_file_and_series_directory(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            series = root_path / "series"
            series.mkdir()
            instance = series / "level-0.dcm"
            instance.touch()
            self.assertTrue(hx.util.is_dicom_slide(instance))
            self.assertTrue(hx.util.is_dicom_slide(series))

    def test_get_slide_paths_groups_series_directory(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            series = root_path / "series"
            series.mkdir()
            (series / "level-0.dcm").touch()
            (series / "level-1.dcm").touch()
            (root_path / "other.svs").touch()
            paths = hx.util.get_slide_paths(root)
            self.assertIn(str(series), paths)
            self.assertIn(str(root_path / "other.svs"), paths)
            self.assertNotIn(str(series / "level-0.dcm"), paths)
            self.assertNotIn(str(series / "level-1.dcm"), paths)


class TestDicomReader(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.series = Path(self.tempdir.name)
        (self.series / "level-0.dcm").touch()
        fake_module = types.ModuleType("wsidicom")
        fake_module.WsiDicom = _FakeWsiDicom
        self.module_patch = mock.patch.dict(sys.modules, {"wsidicom": fake_module})
        self.module_patch.start()

    def tearDown(self):
        self.module_patch.stop()
        self.tempdir.cleanup()

    def test_metadata_and_levels(self):
        reader = _DicomReader(str(self.series))
        self.assertEqual(reader.dimensions, (1000, 800))
        self.assertEqual(reader.mpp, 0.25)
        self.assertEqual(reader.level_dimensions, [(1000, 800), (250, 200)])
        self.assertEqual(reader.level_downsamples, [1.0, 4.0])
        self.assertEqual(reader.best_level_for_downsample(4.0), 1)

    def test_region_padding_resize_and_thumbnail(self):
        reader = _DicomReader(str(self.series))
        region = reader.read_region((-10, -20), 0, (64, 64))
        self.assertEqual(region.shape, (64, 64, 3))
        self.assertTrue(region.flags.writeable)
        self.assertTrue(np.all(region[:20] == 255))
        resized = reader.read_from_pyramid((0, 0), (256, 256), (32, 32))
        self.assertEqual(resized.shape, (32, 32, 3))
        thumbnail = reader.thumbnail(width=100)
        self.assertEqual(thumbnail.shape, (80, 100, 3))

    def test_optional_dependency_error_is_actionable(self):
        self.module_patch.stop()
        with mock.patch.dict(sys.modules, {"wsidicom": None}):
            with self.assertRaisesRegex(ImportError, "histox\\[dicom\\]"):
                _DicomReader(str(self.series))
        self.module_patch.start()


if __name__ == "__main__":
    unittest.main()
