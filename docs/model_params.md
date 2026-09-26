# histox.ModelParams

The `ModelParams` class organizes model and training parameters/hyperparameters and assists with model building.

See [Training](training.html#training) for a detailed look at how to train models.

## ModelParams

*class*histox.ModelParams(***, *loss: str = 'CrossEntropy'*, ***kwargs*)[[source]](_modules/histox/model/torch.html#ModelParams)

Build a set of hyperparameters.

ModelParams.to_dict(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

ModelParams.get_normalizer(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

ModelParams.validate(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

ModelParams.model_type(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

## Mini-batch balancing

During training, mini-batch balancing can be customized to assist with increasing representation of sparse outcomes or small slides. Five mini-batch balancing methods are available when configuring `histox.ModelParams`, set through the parameters `training_balance` and `validation_balance`. These are `'tile'`, `'category'`, `'patient'`, `'slide'`, and `'none'`.

If **tile-level balancing** ("tile") is used, tiles will be selected randomly from the population of all extracted tiles.

If **slide-based balancing** ("patient") is used, batches will contain equal representation of images from each slide.

If **patient-based balancing** ("patient") is used, batches will balance image tiles across patients. The balancing is similar to slide-based balancing, except across patients (as each patient may have more than one slide).

If **category-based balancing** ("category") is used, batches will contain equal representation from each outcome category.

If **no balancing** is performed, batches will be assembled by randomly selecting from TFRecords. This is equivalent to slide-based balancing if each slide has its own TFRecord (default behavior).

See [Oversampling with balancing](dataloaders.html#balancing) for more discussion on sampling and mini-batch balancing.

Note

If you are [using a Trainer](training.html#training-with-trainer) to train your models, you can further customize the mini-batch balancing strategy by using `histox.Dataset.balance()` on your training and/or validation datasets.