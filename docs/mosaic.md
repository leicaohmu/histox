# histox.Mosaic

`histox.Mosaic` plots tile images onto a map of slides, generating a mosaic map.

The idea of a mosaic map is to visualize image feature variation across slides and among categories, in an attempt
to better understand the kinds of image features discriminative models might be using to generate class predictions.
They are created by first generating whole-dataset layer features (using
[`histox.DatasetFeatures`](dataset_features.html#histox.DatasetFeatures)), which are then mapped into two-dimensional space using UMAP
dimensionality reduction ([`histox.SlideMap`](slidemap.html#histox.SlideMap)). This resulting SlideMap is then passed to
`histox.Mosaic`, which overlays tile images onto the dimensionality-reduced slide map.

An example of a mosaic map can be found in Figure 4 of [this paper](https://doi.org/10.1038/s41379-020-00724-3).
It bears some resemblence to the Activation Atlases created by
[Google and OpenAI](https://distill.pub/2019/activation-atlas/), without the use of feature inversion.

See [Mosaic Maps](posthoc.html#mosaic-map) for an example of how a mosaic map can be used in the context of a project.

*class*histox.Mosaic(*images: [SlideMap](slidemap.html#histox.SlideMap) | List[ndarray] | ndarray | List[Tuple[str, int]]*, *coords: Tuple[int, int] | ndarray | None = None*, ***, *tfrecords: List[str] = None*, *normalizer: str | [StainNormalizer](norm.html#histox.norm.StainNormalizer) | None = None*, *normalizer_source: str | None = None*, ***grid_kwargs*)[[source]](_modules/histox/mosaic.html#Mosaic)

Visualization of plotted image tiles.

## Methods

Mosaic.generate_grid(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

Mosaic.plot(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

Mosaic.points_at_grid_index(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

Mosaic.save(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

Mosaic.save_report(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

Mosaic.view(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.