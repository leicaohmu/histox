import io
import logging
import os
import tempfile
import unittest
from unittest import mock

import histox as hx
from histox.util.log_utils import MultiProcessingHandler


class MultiProcessingHandlerTest(unittest.TestCase):

    def setUp(self):
        self.stream = io.StringIO()
        self.sub_handler = logging.StreamHandler(self.stream)
        self.sub_handler.setFormatter(logging.Formatter('%(levelname)s:%(message)s'))
        self.handler = MultiProcessingHandler('histox-log-test', self.sub_handler)
        self.logger = logging.getLogger(f'{__name__}.{id(self)}')
        self.logger.handlers = [self.handler]
        self.logger.propagate = False
        self.logger.setLevel(logging.DEBUG)

    def tearDown(self):
        self.logger.handlers = []
        self.handler.close()

    def test_close_drains_queued_records(self):
        self.logger.info('queued message')

        self.handler.close()

        self.assertEqual(self.stream.getvalue(), 'INFO:queued message\n')

    def test_exception_record_is_formatted(self):
        try:
            raise ValueError('example error')
        except ValueError:
            self.logger.exception('operation failed')

        self.handler.close()

        output = self.stream.getvalue()
        self.assertIn('ERROR:operation failed', output)
        self.assertIn('ValueError: example error', output)

    def test_close_is_idempotent_and_stops_listener(self):
        listener_thread = self.handler._listener._thread

        self.handler.close()
        self.handler.close()

        self.assertFalse(listener_thread.is_alive())


class LoggingConfigurationTest(unittest.TestCase):

    def _assert_file_handler_writes(self, handler, path):
        try:
            record = logging.LogRecord(
                name='histox',
                level=logging.INFO,
                pathname=__file__,
                lineno=0,
                msg='explicit file logging',
                args=(),
                exc_info=None,
            )
            handler.emit(record)
            handler.close()

            with open(path, encoding='utf-8') as log_file:
                self.assertIn('explicit file logging', log_file.read())
        finally:
            hx.log.removeHandler(handler)
            handler.close()

    def test_file_logging_is_disabled_by_default(self):
        file_handlers = [
            handler for handler in hx.log.handlers
            if isinstance(
                getattr(handler, 'sub_handler', None),
                logging.FileHandler,
            )
        ]

        self.assertEqual(file_handlers, [])

    def test_file_logging_can_be_enabled_programmatically(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, 'histox.log')

            handler = hx.util.addLoggingFileHandler(path)

            self._assert_file_handler_writes(handler, path)

    def test_file_logging_can_be_enabled_from_environment(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, 'histox.log')
            with mock.patch.dict(os.environ, {'HX_LOG_FILE': path}):
                handler = hx.util._configure_file_logging()

            self.assertIsNotNone(handler)
            self._assert_file_handler_writes(handler, path)


if __name__ == '__main__':
    unittest.main()
