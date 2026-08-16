"""Threaded execution over molecule-sized chunks of segment- or atom-level
arrays.

Each pass splits *molecules* (not segments or atoms) into contiguous
ranges so threads own disjoint molecules and need no locking.
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

    Chunks cover disjoint molecules, so ``fn`` may write only to its own
    ``[start, stop)`` range.

    Parameters
    ----------
    fn : Callable[[int, int], None]
        Called once per chunk with that chunk's molecule bounds.
        Exceptions in a thread are re-raised in the caller.
    num_items : int
        Number of molecules to split across threads.
    num_threads : int | None, optional
        Thread count. ``None`` (default) uses every CPU core.
    limit_blas : bool, optional
        If True, cap BLAS to 1 thread for the duration of the call, so
        numpy matrix products do not oversubscribe the CPU.
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
