'''This module contains low-level utilies for interfacing with TFRecords.

Code from this module was inspired in part by https://github.com/vahidk/tfrecord.
'''

from histox.tfrecord import \
    iterator_utils  # noqa # pylint: disable=unused-import
from histox.tfrecord import reader  # noqa # pylint: disable=unused-import
from histox.tfrecord import tools  # noqa # pylint: disable=unused-import
from histox.tfrecord import writer  # noqa # pylint: disable=unused-import
from histox.tfrecord.iterator_utils import *  # noqa # pylint: disable=unused-import
from histox.tfrecord.reader import *  # noqa # pylint: disable=unused-import
from histox.tfrecord.writer import *  # noqa # pylint: disable=unused-import
from histox.util import example_pb2  # noqa # pylint: disable=unused-import
