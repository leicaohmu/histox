# Developer notes

These notes cover extension points, data-loading internals, custom feature
extractors, plugins, and debugging. They are intended for contributors and for
researchers building components beyond the high-level user guide.

- [Extensions](extensions.html)
- [TFRecords: Reading and Writing](tfrecords.html)
- [Dataloaders: Sampling and Augmentation](dataloaders.html)
- [Custom Feature Extractors](custom_extractors.html)
- [Strong Supervision with Tile Labels](tile_labels.html)
- [Creating a HistoX Plugin](plugins.html)
- [Troubleshooting](troubleshooting.html)

## Compatibility APIs

TensorFlow compatibility remains documented while HistoX moves toward a
PyTorch-first public interface.

- [histox.io.tensorflow](io_tensorflow.html)
- [histox.model.tensorflow](model_tensorflow.html)