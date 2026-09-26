# histox.slide

This module contains classes to load slides and extract tiles. For optimal performance, tile extraction should
generally not be performed by instancing these classes directly, but by calling either
`histox.Project.extract_tiles()` or `histox.Dataset.extract_tiles()`, which include performance
optimizations and additional functionality.

## histox.WSI

*class*histox.slide.WSI(*path: str*, *tile_px: int*, *tile_um: int | str*, *stride_div: int = 1*, ***, *enable_downsample: bool = True*, *roi_dir: str | None = None*, *rois: List[str] | None = None*, *roi_method: str = 'auto'*, *roi_filter_method: str | float = 'center'*, *origin: str | Tuple[int, int] = (0, 0)*, *pb: Progress | None = None*, *verbose: bool = True*, *use_edge_tiles: bool = False*, *mpp: float | None = None*, *simplify_roi_tolerance: float | None = None*, *artifact_labels: List[str] | None = None*, *disable_grid: bool = False*, ***reader_kwargs: Any*)[[source]](_modules/histox/slide/wsi.html#WSI)

Loads a slide and its annotated region of interest (ROI).

### Attributes

| `WSI.dimensions` | Dimensions of highest-magnification level (width, height) |
| --- | --- |
| `WSI.qc_mask` | Returns union of all QC masks. |
| `WSI.levels` | List of dict, with metadata for each level. |
| `WSI.level_dimensions` | List of list, with dimensions for each slide level. |
| `WSI.level_downsamples` | Downsample of each level (starts at 1, increases with lower mag). |
| `WSI.level_mpp` | Microns-per-pixel (MPP) for each level. |
| `WSI.properties` | Dictionary of metadata loaded from the slide. |
| `WSI.slide` | Backend-specific slide object. |
| `WSI.vendor` | Slide scanner vendor, if available. |

### Methods

WSI.align_to(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

WSI.align_tiles_to(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

WSI.apply_qc_mask(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

WSI.apply_segmentation(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

WSI.area(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

WSI.build_generator(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

WSI.dim_to_mpp(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

WSI.get_tile_mask(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

WSI.get_tile_dataframe(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

WSI.extract_cells(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

WSI.extract_tiles(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

WSI.export_rois(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

WSI.has_rois(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

WSI.load_csv_roi(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

WSI.load_json_roi(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

WSI.load_roi_array(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

WSI.mpp_to_dim(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

WSI.predict(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

WSI.preview(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

WSI.process_rois(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

WSI.show_alignment(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

WSI.square_thumb(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

WSI.qc(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

WSI.remove_qc(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

WSI.remove_roi(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

WSI.tensorflow(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

WSI.torch(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

WSI.thumb(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

WSI.verify_alignment(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

WSI.view(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

## Other functions

slide.predict(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.