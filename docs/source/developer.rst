Developer notes
===============

These notes cover extension points, data-loading internals, custom feature
extractors, plugins, and debugging. They are intended for contributors and for
researchers building components beyond the high-level user guide.

.. toctree::
   :maxdepth: 1

   extensions
   tfrecords
   dataloaders
   custom_extractors
   tile_labels
   plugins
   troubleshooting

Compatibility APIs
------------------

TensorFlow compatibility remains documented while HistoX moves toward a
PyTorch-first public interface.

.. toctree::
   :maxdepth: 1

   io_tensorflow
   model_tensorflow
