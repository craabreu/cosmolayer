"""Threaded, molecule-chunked execution shared by every store operation
that walks the segment- or atom-level arrays one molecule at a time.

Every threaded pass in this package splits the *molecules* (never
segments or atoms) into ``num_threads`` contiguous ranges, so each thread
owns disjoint molecules and needs no locking around the accumulation
arrays it writes into.
"""

import os
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext

import threadpoolctl


def resolve_num_threads(num_threads: int | None) -> int:
    """Resolve a possibly-unset thread count to a concrete positive value.

    Parameters
    ----------
    num_threads : int | None
        Requested number of threads, or None to mean "every available CPU
        core".

    Returns
    -------
    int
        ``num_threads`` if given, else ``os.cpu_count()`` (falling back to
        1 if that returns None, e.g. in a constrained container).
    """
    if num_threads is not None:
        return num_threads
    return os.cpu_count() or 1


def molecule_chunks(num_items: int, num_threads: int) -> Iterator[tuple[int, int]]:
    """Split ``range(num_items)`` into up to ``num_threads`` contiguous,
    roughly equal chunks.

    Parameters
    ----------
    num_items : int
        Total number of molecules (or other items) to split.
    num_threads : int
        Number of chunks to aim for. Fewer are yielded if
        ``num_items < num_threads``.

    Yields
    ------
    tuple[int, int]
        ``(start, stop)`` bounds of each chunk, usable as a Python slice.
    """
    if num_items == 0:
        return
    chunk_size = (num_items + num_threads - 1) // num_threads
    for start in range(0, num_items, chunk_size):
        yield start, min(start + chunk_size, num_items)


def run_in_threads(
    fn: Callable[[int, int], None],
    num_items: int,
    *,
    num_threads: int | None = None,
    limit_blas: bool = False,
) -> None:
    """Run ``fn(start, stop)`` once per molecule chunk, across threads.

    Every chunk covers disjoint molecules, so ``fn`` is safe to call
    concurrently as long as it only ever writes to the rows/segments
    belonging to its own ``[start, stop)`` range.

    Parameters
    ----------
    fn : Callable[[int, int], None]
        Called once per chunk with that chunk's ``(start, stop)`` molecule
        bounds (see ``molecule_chunks``). Any exception raised in a thread
        is re-raised in the calling thread.
    num_items : int
        Total number of molecules to split across threads.
    num_threads : int | None, optional
        Number of threads to use, by default None, meaning every available
        CPU core (see ``resolve_num_threads``).
    limit_blas : bool, optional
        Whether to cap BLAS's own internal thread pool to 1 thread for the
        duration of the call, by default False. Set this when ``fn`` calls
        into BLAS-backed numpy operations (e.g. matrix products) that would
        otherwise each spawn their own thread pool, oversubscribing the
        CPU on top of this function's own threading.

    Returns
    -------
    None
    """
    num_threads = resolve_num_threads(num_threads)
    limiter = threadpoolctl.threadpool_limits(limits=1) if limit_blas else nullcontext()
    with limiter, ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(fn, start, stop)
            for start, stop in molecule_chunks(num_items, num_threads)
        ]
        for future in as_completed(futures):
            future.result()
