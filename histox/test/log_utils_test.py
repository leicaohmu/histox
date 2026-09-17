import io
import logging
import unittest

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


if __name__ == '__main__':
    unittest.main()
