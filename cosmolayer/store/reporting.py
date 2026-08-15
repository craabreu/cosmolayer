"""Console reporting helpers for the ``cosmolayer.store`` CLI."""

from typing import Any

import numpy as np
from numpy.typing import NDArray


def print_stats(
    title: str,
    properties: NDArray[np.number[Any]],
    quantiles: tuple[float, ...] = (0.01, 0.1, 0.5, 0.9, 0.99),
    value_format: str = ".6f",
) -> None:
    """Print min, max, mean, and quantiles of a property as a table.

    Parameters
    ----------
    title : str
        Label identifying the property, used as the table's column header.
    properties : np.ndarray
        Values to summarize, of any numeric dtype (integer or floating).
    quantiles : tuple[float, ...], optional
        Quantiles (in ``[0, 1]``) to report, by default
        ``(0.01, 0.1, 0.5, 0.9, 0.99)``.
    value_format : str, optional
        Format spec for the reported values, by default ``".6f"``. Use an
        exponential format such as ``".3e"`` for quantities whose whole
        range sits far below the fixed-point resolution, which would
        otherwise print as a column of zeros.

    Returns
    -------
    None
        The statistics are printed directly; nothing is returned.
    """
    print(f"\n{'Statistic':<10} | {title:>20}")
    print("-" * 33)
    print(f"{'Count':<10} | {len(properties):>20}")
    print(f"{'Min':<10} | {properties.min():>20{value_format}}")
    for q in sorted(quantiles):
        q_percent = q * 100
        if int(q_percent) == q_percent:
            qtext = f"{q_percent:.0f}%"
        else:
            qtext = f"{q_percent:.1f}%"
        print(f"{qtext:<10} | {np.quantile(properties, q):>20{value_format}}")
    print(f"{'Max':<10} | {properties.max():>20{value_format}}")
    print(f"{'Mean':<10} | {properties.mean():>20{value_format}}")
    print()
