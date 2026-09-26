.. currentmodule:: histox.io.torch

histox.io.torch
===================

The purpose of this module is to provide a performant, backend-agnostic TFRecord reader and interleaver to use as
input for PyTorch models. Its TFRecord reader is a modified and optimized version of
https://github.com/vahidk/tfrecord, included as the module :mod:`histox.tfrecord`. TFRecord file reading and
interleaving is supervised by :func:`histox.io.torch.interleave`, while the
:func:`histox.io.torch.interleave_dataloader` function provides a PyTorch DataLoader object which can be directly used.

.. automodule:: histox.io.torch
    :members:
    :exclude-members: StyleGAN2Interleaver, TileLabelInterleaver, InterleaveIterator, IndexedInterleaver

.. autoclass:: histox.io.torch.InterleaveIterator

.. autoclass:: histox.io.torch.IndexedInterleaver