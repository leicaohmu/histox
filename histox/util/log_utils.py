import logging
import queue
from logging.handlers import QueueHandler, QueueListener
from tqdm import tqdm


class LogFormatter(logging.Formatter):
    MSG_FORMAT = "%(message)s"
    LEVEL_FORMATS = {
        logging.DEBUG: f"[dim]{MSG_FORMAT}[/]",
        logging.INFO: MSG_FORMAT,
        logging.WARNING: f"[yellow]{MSG_FORMAT}[/]",
        logging.ERROR: f"[red]{MSG_FORMAT}[/]",
        logging.CRITICAL: f"[red bold]{MSG_FORMAT}[/]"
    }

    def format(self, record):
        log_fmt = self.LEVEL_FORMATS[record.levelno]
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class FileFormatter(logging.Formatter):
    MSG_FORMAT = "%(asctime)s [%(levelname)s] - %(message)s"
    FORMAT_CHARS = ['\033[1m', '\033[2m', '\033[4m', '\033[91m', '\033[92m',
                    '\033[93m', '\033[94m', '\033[38;5;5m', '\033[0m']

    def format(self, record):
        formatter = logging.Formatter(
            fmt=self.MSG_FORMAT,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        formatted = formatter.format(record)
        for char in self.FORMAT_CHARS:
            formatted = formatted.replace(char, '')
        return formatted


class MultiProcessingHandler(logging.Handler):
    """Forward records asynchronously to another logging handler.

    The public name is retained for compatibility. This handler uses a
    thread-backed queue and does not configure logging in child processes.
    """

    def __init__(self, name, sub_handler=None):
        super().__init__()

        if sub_handler is None:
            sub_handler = logging.StreamHandler()
        self.sub_handler = sub_handler
        self.setLevel(self.sub_handler.level)
        self.setFormatter(self.sub_handler.formatter)
        self.queue = queue.Queue()
        self._queue_handler = QueueHandler(self.queue)
        self._listener = QueueListener(
            self.queue,
            self.sub_handler,
            respect_handler_level=True,
        )
        self._is_closed = False
        self._listener.start()

    def setFormatter(self, fmt):
        super().setFormatter(fmt)
        self.sub_handler.setFormatter(fmt)

    def emit(self, record):
        self._queue_handler.emit(record)

    def close(self):
        if not self._is_closed:
            self._is_closed = True
            self._listener.stop()
            self.sub_handler.close()
            super().close()


class TqdmLoggingHandler(logging.StreamHandler):
    """Avoid tqdm progress bar interruption by logger's output to console"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flush_line = False

    def emit(self, record):
        try:
            msg = self.format(record)
            if self.flush_line:
                msg = '\r\033[K' + msg
            tqdm.write(msg, end=self.terminator)
        except RecursionError:
            raise
        except Exception:
            print(f"problems with msg {record}")
            self.handleError(record)
