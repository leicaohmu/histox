# histox.DatasetFeatures

*class*histox.DatasetFeatures(*model: str | tf.keras.models.Model | torch.nn.Module*, *dataset: hx.Dataset*, ***, *labels: Dict[str, str] | Dict[str, int] | Dict[str, List[float]] | None = None*, *cache: str | None = None*, *annotations: Dict[str, str] | Dict[str, int] | Dict[str, List[float]] | None = None*, ***kwargs: Any*)[[source]](_modules/histox/model/features.html#DatasetFeatures)

Loads annotations, saved layer activations / features, and prepares
output saving directories. Will also read/write processed features to a
PKL cache file to save time in future iterations.

Note

Storing predictions along with layer features is optional, to offer the user
reduced memory footprint. For example, saving predictions for a 10,000 slide
dataset with 1000 categorical outcomes would require:

4 bytes/float32-logit
* 1000 predictions/slide
* 3000 tiles/slide
* 10000 slides
~= 112 GB

## Methods

DatasetFeatures.activations_by_category(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

DatasetFeatures.box_plots(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

DatasetFeatures.concat(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

DatasetFeatures.from_df(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

DatasetFeatures.load_cache(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

DatasetFeatures.map_activations(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

DatasetFeatures.map_predictions(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

DatasetFeatures.merge(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

DatasetFeatures.remove_slide(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

DatasetFeatures.save_cache(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

DatasetFeatures.save_example_tiles(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

DatasetFeatures.softmax_mean(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

DatasetFeatures.softmax_percent(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

DatasetFeatures.softmax_predict(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

DatasetFeatures.stats(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

DatasetFeatures.to_csv(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

DatasetFeatures.to_df(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

DatasetFeatures.to_torch(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.