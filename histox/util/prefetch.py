"""Sequential read-ahead of slide files into the OS page cache.

Whole-slide images are usually read as many small, scattered tile requests.
On spinning disks and many network filesystems that access pattern is
dominated by seek time: on a two-disk HDD array, reading 6 TCGA slides
(7.2 GB) tile by tile ran at about 27 MB/s, while a plain sequential read of
the same files ran at 149 MB/s. Reading a slide once, front to back, before
its tiles are requested lets the tile reader hit the page cache instead.

Prefetching is skipped for files larger than a fraction of the currently
available memory, so it never pushes the system into swap. Set the
environment variable ``HX_PREFETCH=0`` to disable it globally.
"""

import logging
import os
import threading
from typing import Optional, Sequence

log = logging.getLogger('histox')

_CHUNK = 16 << 20  # 16 MiB


def prefetch_enabled() -> bool:
    """Whether prefetching is enabled (environment variable ``HX_PREFETCH``)."""
    return os.environ.get('HX_PREFETCH', '1').strip().lower() not in (
        '0', 'false', 'no', 'off')


def _available_memory() -> Optional[int]:
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def should_prefetch(path: str, max_fraction: float = 0.5) -> bool:
    """Whether ``path`` is a regular file small enough to prefetch."""
    if not prefetch_enabled():
        return False
    try:
        if not os.path.isfile(path):
            return False
        size = os.path.getsize(path)
    except OSError:
        return False
    available = _available_memory()
    return available is None or size <= available * max_fraction


def prefetch_file(
    path: str,
    max_fraction: float = 0.5,
    stop: Optional[threading.Event] = None
) -> bool:
    """Read a file sequentially so that it lands in the OS page cache.

    Args:
        path (str): File to read.
        max_fraction (float): Skip files larger than this fraction of the
            currently available memory. Defaults to 0.5.
        stop (threading.Event, optional): Abort early when set.

    Returns:
        bool: True if the whole file was read.
    """
    if not should_prefetch(path, max_fraction):
        return False
    try:
        buf = bytearray(_CHUNK)
        with open(path, 'rb', buffering=0) as f:
            while f.readinto(buf):
                if stop is not None and stop.is_set():
                    return False
    except OSError as e:
        log.debug(f"Prefetch of {path} failed: {e}")
        return False
    return True


class SlidePrefetcher:
    """Prefetch a list of slides, staying one slide ahead of the consumer.

    A background thread reads slide ``i + 1`` while the caller processes
    slide ``i``. Call :meth:`wait` with each index, in order, before reading
    that slide. Use as a context manager.

    .. code-block:: python

        with SlidePrefetcher(paths) as prefetcher:
            for i, path in enumerate(paths):
                prefetcher.wait(i)
                process(path)

    Args:
        paths (list(str)): Slide paths, in the order they will be processed.
        enabled (bool): If False, :meth:`wait` returns immediately and
            nothing is read. Defaults to True.
        max_fraction (float): See :func:`prefetch_file`.
    """

    def __init__(
        self,
        paths: Sequence[str],
        enabled: bool = True,
        max_fraction: float = 0.5
    ) -> None:
        self.paths = [str(p) for p in paths]
        self.enabled = bool(enabled) and prefetch_enabled() and len(self.paths) > 0
        self.max_fraction = max_fraction
        self._ready = [threading.Event() for _ in self.paths]
        self._started = [threading.Event() for _ in self.paths]
        self._stop = threading.Event()
        self._thread = None  # type: Optional[threading.Thread]

    def __enter__(self) -> "SlidePrefetcher":
        if self.enabled:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        else:
            for event in self._ready:
                event.set()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _run(self) -> None:
        try:
            for i, path in enumerate(self.paths):
                if i > 0:
                    # Stay at most one slide ahead, so prefetched data is not
                    # evicted before it is used.
                    while not self._started[i - 1].wait(0.2):
                        if self._stop.is_set():
                            return
                if self._stop.is_set():
                    return
                prefetch_file(path, self.max_fraction, stop=self._stop)
                self._ready[i].set()
        finally:
            for event in self._ready:
                event.set()

    def wait(self, index: int) -> None:
        """Mark slide ``index`` as started and wait until it is prefetched."""
        self._started[index].set()
        self._ready[index].wait()

    def close(self) -> None:
        """Stop the background thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
