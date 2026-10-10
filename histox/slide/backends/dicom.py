"""DICOM Slide Microscopy reader backed by :mod:`wsidicom`.

The dependency is optional. Install it with ``pip install histox[dicom]``.
Local DICOM WSI pyramids are commonly stored as a directory containing one
``.dcm`` file per pyramid level; pass that directory to :class:`histox.WSI`.
"""

import os

import cv2
import numpy as np

from pathlib import Path
from skimage.color import rgb2hsv
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union

from histox.util import log
from histox.slide.utils import *


SUPPORTED_BACKEND_FORMATS = ["dcm"]

__dicom_reader__ = None
__dicom_reader_path__ = None
__dicom_reader_args__ = None
__dicom_reader_kwargs__ = None


def get_dicom_reader(path: str, *args, **kwargs):
    """Return a cached DICOM reader for ``path``."""
    global __dicom_reader__, __dicom_reader_path__
    global __dicom_reader_args__, __dicom_reader_kwargs__

    if (
        __dicom_reader_path__ == path
        and __dicom_reader_args__ == args
        and __dicom_reader_kwargs__ == kwargs
    ):
        return __dicom_reader__

    reader = _DicomReader(path, *args, **kwargs)
    __dicom_reader_path__ = path
    __dicom_reader_args__ = args
    __dicom_reader_kwargs__ = kwargs
    __dicom_reader__ = reader
    return reader


def tile_worker(
    c: List[int],
    args: SimpleNamespace,
) -> Optional[Union[str, Dict]]:
    """Extract one tile from a DICOM Slide Microscopy pyramid."""
    if args.has_segmentation:
        c, tile_mask = c
        (x, y, grid_x), grid_y = c, 0
    else:
        tile_mask = None
        x, y, grid_x, grid_y = c

    x_coord = int(x + args.full_extract_px / 2)
    y_coord = int(y + args.full_extract_px / 2)
    slide = get_dicom_reader(
        args.path,
        args.mpp_override,
        **args.reader_kwargs,
    )

    if args.whitespace_fraction < 1 or args.grayspace_fraction < 1:
        if args.filter_downsample_ratio > 1:
            filter_extract_px = args.extract_px // args.filter_downsample_ratio
            filter_region = slide.read_region(
                (x, y),
                args.filter_downsample_level,
                (filter_extract_px, filter_extract_px),
            )
        else:
            filter_region = slide.read_region(
                (x, y),
                args.downsample_level,
                (args.extract_px, args.extract_px),
            )
        if args.whitespace_fraction < 1:
            ws_fraction = np.mean(
                np.mean(filter_region, axis=-1) > args.whitespace_threshold
            )
            if (
                ws_fraction > args.whitespace_fraction
                and args.whitespace_fraction != FORCE_CALCULATE_WHITESPACE
            ):
                return None
        if args.grayspace_fraction < 1:
            hsv_region = rgb2hsv(filter_region)
            gs_fraction = np.mean(
                hsv_region[:, :, 1] < args.grayspace_threshold
            )
            if (
                gs_fraction > args.grayspace_fraction
                and args.whitespace_fraction != FORCE_CALCULATE_WHITESPACE
            ):
                return None

    return_dict = {
        "loc": [x_coord, y_coord],
        "grid": [grid_x, grid_y],
    }
    if args.grayspace_fraction < 1:
        return_dict["gs_fraction"] = gs_fraction
    if args.whitespace_fraction < 1:
        return_dict["ws_fraction"] = ws_fraction
    if args.dry_run:
        return return_dict

    if tile_mask is not None:
        tile_mask = cv2.resize(
            tile_mask,
            (args.tile_px, args.tile_px),
            interpolation=cv2.INTER_NEAREST,
        )

    try:
        region = slide.read_region(
            (x, y),
            args.downsample_level,
            (args.extract_px, args.extract_px),
        )
    except Exception as error:
        log.warning(f"Error reading DICOM region at ({x}, {y}): {error}")
        return None

    if int(args.tile_px) != int(args.extract_px):
        region = cv2.resize(region, (args.tile_px, args.tile_px))
    if region.shape[-1] == 4:
        region = region[:, :, :3]
    if tile_mask is not None:
        region[tile_mask == 0] = (0, 0, 0)

    if args.normalizer:
        try:
            region = args.normalizer.rgb_to_rgb(region)
        except Exception:
            return None

    if args.img_format != "numpy":
        if args.img_format not in ("jpg", "jpeg", "png"):
            raise ValueError(f"Unknown image format {args.img_format}")
        bgr_region = cv2.cvtColor(region, cv2.COLOR_RGB2BGR)
        image = cv2.imencode("." + args.img_format, bgr_region)[1].tobytes()
    else:
        image = region

    if args.yolo or args.draw_roi:
        coords, boxes, yolo_anns = roi_coords_from_image(c, args)
    if args.draw_roi:
        image = draw_roi(image, coords)
    return_dict["image"] = image
    if args.yolo:
        return_dict["yolo"] = yolo_anns
    return return_dict


class _DicomReader:
    """Adapter from ``wsidicom.WsiDicom`` to the HistoX reader contract."""

    has_levels = True

    def __init__(
        self,
        path: str,
        mpp: Optional[float] = None,
        *,
        cache_kw: Optional[Dict[str, Any]] = None,
        ignore_missing_mpp: bool = False,
        pad_missing: bool = True,
        use_bounds: bool = False,
        transforms: Optional[List[int]] = None,
        num_workers: int = 1,
    ) -> None:
        try:
            from wsidicom import WsiDicom
        except ImportError as error:
            raise ImportError(
                "DICOM Slide Microscopy support requires the optional "
                "'wsidicom' dependency. Install it with "
                "`pip install histox[dicom]`."
            ) from error

        if use_bounds:
            raise ValueError("use_bounds is not supported for DICOM slides")
        if transforms:
            raise ValueError("transforms are not supported for DICOM slides")
        if not os.path.isdir(path) and Path(path).suffix.lower() != ".dcm":
            raise ValueError(
                "DICOM input must be a .dcm file or a directory containing "
                "one DICOM Slide Microscopy series"
            )

        self.path = path
        self.pad_missing = pad_missing
        self.cache_kw = cache_kw or {}
        self.num_workers = max(1, int(num_workers))
        self.reader = WsiDicom.open(path)
        self.dimensions = (
            int(self.reader.size.width),
            int(self.reader.size.height),
        )

        detected_mpp_x = float(self.reader.mpp.width)
        detected_mpp_y = float(self.reader.mpp.height)
        if mpp is not None:
            self._mpp = float(mpp)
        elif detected_mpp_x > 0:
            self._mpp = detected_mpp_x
        elif ignore_missing_mpp:
            self._mpp = DEFAULT_JPG_MPP
        else:
            raise ValueError(f"DICOM slide {path} does not define pixel spacing")
        if not np.isclose(detected_mpp_x, detected_mpp_y):
            log.warning(
                "DICOM slide has anisotropic pixel spacing "
                f"({detected_mpp_x}, {detected_mpp_y}); HistoX will use "
                f"the x-axis value {self._mpp}."
            )

        self.properties = {
            OPS_VENDOR: "dicom",
            OPS_MPP_X: self._mpp,
            "dicom.mpp-x": detected_mpp_x,
            "dicom.mpp-y": detected_mpp_y,
            "dicom.path": path,
        }
        self.levels = []
        base_mpp = float(self.reader.levels[0].mpp.width)
        for list_index, level in enumerate(self.reader.levels):
            width = int(level.size.width)
            height = int(level.size.height)
            downsample = float(level.mpp.width) / base_mpp
            self.levels.append(
                {
                    "dimensions": (width, height),
                    "width": width,
                    "height": height,
                    "downsample": downsample,
                    "level": list_index,
                    "dicom_level": int(level.level),
                }
            )
        self.level_count = len(self.levels)
        self.level_dimensions = [level["dimensions"] for level in self.levels]
        self.level_downsamples = [level["downsample"] for level in self.levels]

    @property
    def mpp(self) -> float:
        return self._mpp

    def has_mpp(self) -> bool:
        return self._mpp is not None

    def best_level_for_downsample(self, downsample: float) -> int:
        eligible = [
            index
            for index, value in enumerate(self.level_downsamples)
            if value <= downsample
        ]
        return eligible[-1] if eligible else 0

    def coord_to_raw(self, x, y):
        return x, y

    def raw_to_coord(self, x, y):
        return x, y

    def _read_region_array(
        self,
        base_level_dim: Tuple[int, int],
        downsample_level: int,
        extract_size: Tuple[int, int],
        pad_missing: bool,
    ) -> np.ndarray:
        level = self.levels[downsample_level]
        downsample = level["downsample"]
        x = int(np.floor(base_level_dim[0] / downsample))
        y = int(np.floor(base_level_dim[1] / downsample))
        width, height = (int(extract_size[0]), int(extract_size[1]))
        level_width, level_height = level["dimensions"]

        read_x = max(0, x)
        read_y = max(0, y)
        read_width = min(level_width, x + width) - read_x
        read_height = min(level_height, y + height) - read_y
        if read_width <= 0 or read_height <= 0:
            if not pad_missing:
                raise ValueError("Requested DICOM region is outside the slide")
            return np.full((height, width, 3), 255, dtype=np.uint8)

        image = self.reader.read_region(
            (read_x, read_y),
            level["dicom_level"],
            (read_width, read_height),
            threads=self.num_workers,
        ).convert("RGB")
        region = np.asarray(image, dtype=np.uint8).copy()
        if (
            not pad_missing
            or (read_x == x and read_y == y
                and read_width == width and read_height == height)
        ):
            return region

        padded = np.full((height, width, 3), 255, dtype=np.uint8)
        offset_x = max(0, -x)
        offset_y = max(0, -y)
        padded[
            offset_y : offset_y + read_height,
            offset_x : offset_x + read_width,
        ] = region
        return padded

    def read_region(
        self,
        base_level_dim: Tuple[int, int],
        downsample_level: int,
        extract_size: Tuple[int, int],
        *,
        convert: Optional[str] = None,
        flatten: bool = False,
        resize_factor: Optional[float] = None,
        pad_missing: Optional[bool] = None,
    ) -> Union[np.ndarray, bytes]:
        del flatten
        should_pad = self.pad_missing if pad_missing is None else pad_missing
        region = self._read_region_array(
            base_level_dim,
            downsample_level,
            extract_size,
            should_pad,
        )
        if resize_factor is not None:
            target = (
                max(1, int(round(region.shape[1] * resize_factor))),
                max(1, int(round(region.shape[0] * resize_factor))),
            )
            region = cv2.resize(region, target)
        if convert and convert.lower() in ("jpg", "jpeg"):
            return numpy2jpg(region)
        if convert and convert.lower() == "png":
            return numpy2png(region)
        if convert not in (None, "numpy"):
            raise ValueError(f"Unsupported conversion format: {convert}")
        return region

    def read_level(self, level: int, to_numpy: bool = False, **kwargs):
        del to_numpy, kwargs
        dimensions = self.level_dimensions[level]
        return self.read_region((0, 0), level, dimensions)

    def read_from_pyramid(
        self,
        top_left: Tuple[int, int],
        window_size: Tuple[int, int],
        target_size: Tuple[int, int],
        *,
        convert: Optional[str] = None,
        flatten: bool = False,
        pad_missing: Optional[bool] = None,
    ) -> Union[np.ndarray, bytes]:
        del flatten
        target_downsample = window_size[0] / target_size[0]
        level = self.best_level_for_downsample(target_downsample)
        downsample = self.level_downsamples[level]
        extract_size = (
            max(1, int(round(window_size[0] / downsample))),
            max(1, int(round(window_size[1] / downsample))),
        )
        region = self.read_region(
            top_left,
            level,
            extract_size,
            pad_missing=pad_missing,
        )
        region = cv2.resize(region, target_size)
        if convert and convert.lower() in ("jpg", "jpeg"):
            return numpy2jpg(region)
        if convert and convert.lower() == "png":
            return numpy2png(region)
        if convert not in (None, "numpy"):
            raise ValueError(f"Unsupported conversion format: {convert}")
        return region

    def thumbnail(
        self,
        width: int = 512,
        level: Optional[int] = None,
        associated: bool = False,
        **kwargs,
    ) -> np.ndarray:
        del level, associated, kwargs
        height = max(1, int(round(width * self.dimensions[1] / self.dimensions[0])))
        return np.asarray(
            self.reader.read_thumbnail((width, height)).convert("RGB"),
            dtype=np.uint8,
        )

    def close(self) -> None:
        self.reader.close()
