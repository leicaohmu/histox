# histox.cellseg

This module contains utility functions for performing whole-slide image cell segmentation with Cellpose.

See [Cell Segmentation](cellseg.html#cellseg) for more information.

histox.cellseg.segment_slide(*slide: [WSI](slide.html#histox.slide.WSI) | str*, *model: cellpose.models.Cellpose | str = 'cyto2'*, ***, *diam_um: float | None = None*, *diam_mean: int | None = None*, *window_size: int | None = None*, *downscale: float | None = None*, *batch_size: int = 8*, *gpus: int | Iterable[int] | None = (0,)*, *spawn_workers: bool = True*, *pb: Progress | None = None*, *pb_tasks: List[TaskID] | None = None*, *show_progress: bool = True*, *save_flow: bool = True*, *cp_thresh: float = 0.0*, *flow_threshold: float = 0.4*, *interp: bool = True*, *tile: bool = True*, *verbose: bool = True*, *device: str | None = None*) → Segmentation[[source]](_modules/histox/cellseg.html#segment_slide)

Segment cells in a whole-slide image, returning masks and centroids.

Parameters:

**slide** (str, `histox.WSI`) - Whole-slide image. May be a path
(str) or WSI object (histox.WSI).

Keyword Arguments:

- **model** (str, `cellpose.models.Cellpose`) - Cellpose model to use
for cell segmentation. May be any valid cellpose model. Defaults
to 'cyto2'.
- **diam_um** (*float**,**optional*) - Cell diameter to detect, in microns.
Determines tile extraction microns-per-pixel resolution to match
the given pixel diameter specified by diam_mean. Not used if
slide is a hx.WSI object.
- **diam_mean** (*int**,**optional*) - Cell diameter to detect, in pixels (without
image resizing). If None, uses Cellpose defaults (17 for the
'nuclei' model, 30 for all others).
- **window_size** (*int*) - Window size, in pixels, at which to segment cells.
Not used if slide is a hx.WSI object.
- **downscale** (*float*) - Factor by which to downscale generated masks after
calculation. Defaults to None (keep masks at original size).
- **batch_size** (*int*) - Batch size for cell segmentation. Defaults to 8.
- **gpus** (*int**,**list**(**int**)*) - GPUs to use for cell segmentation.
Defaults to 0 (first GPU).
- **spawn_workers** (*bool*) - Enable spawn-based multiprocessing. Increases
cell segmentation speed at the cost of higher memory utilization.
- **pb** (`rich.progress.Progress`, optional) - Progress bar instance.
Used for external progress bar tracking. Defaults to None.
- **pb_tasks** (list(`rich.progress.TaskID`)) - Progress bar tasks.
Used for external progress bar tracking. Defaults to None.
- **show_progress** (*bool*) - Show a tqdm progress bar. Defaults to True.
- **save_flow** (*bool*) - Save flow values for the whole-slide image.
Increases memory utilization. Defaults to True.
- **cp_thresh** (*float*) - Cell probability threshold. All pixels with value
above threshold kept for masks, decrease to find more and larger
masks. Defaults to 0.
- **flow_threshold** (*float*) - Flow error threshold (all cells with errors
below threshold are kept). Defaults to 0.4.
- **interp** (*bool*) - Interpolate during 2D dynamics. Defaults to True.
- **tile** (*bool*) - Tiles image to decrease GPU/CPU memory usage.
Defaults to True.
- **verbose** (*bool*) - Verbose log output at the INFO level. Defaults to True.

Returns:

`histox.cellseg.Segmentation`

## Segmentation

*class*histox.cellseg.Segmentation(*masks: ndarray*, ***, *slide: [WSI](slide.html#histox.slide.WSI) | None = None*, *flows: ndarray | None = None*, *styles: ndarray | None = None*, *diams: ndarray | None = None*, *wsi_dim: Tuple[int, int] | None = None*, *wsi_offset: Tuple[int, int] | None = None*)[[source]](_modules/histox/cellseg.html#Segmentation)

cellseg.Segmentation.apply_rois(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

cellseg.Segmentation.calculate_centroids(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

cellseg.Segmentation.calculate_outlines(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

cellseg.Segmentation.centroids(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

cellseg.Segmentation.centroid_to_image(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

cellseg.Segmentation.extract_centroids(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

cellseg.Segmentation.mask_to_image(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

cellseg.Segmentation.outline_to_image(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

cellseg.Segmentation.save(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.