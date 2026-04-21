"""
Bootstrap confidence interval analysis for the elite quadrant count.

This module quantifies sampling variability of the elite quadrant count
(top-25% AWI and PQI) via bootstrap resampling, producing a 95% confidence
interval around the observed point estimate.
"""

import numpy
import pandas as pd


def bootstrap_elite_quadrant(
    df: pd.DataFrame,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Compute a bootstrap confidence interval for the elite quadrant count.

    The elite quadrant is defined as rows where both ``awi_per_minute`` and
    ``mean_pqi`` are at or above their respective 75th percentile thresholds,
    computed from the same DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Player DataFrame containing at least the columns ``awi_per_minute``
        and ``mean_pqi``. Must have at least 4 rows.
    n_bootstrap : int, optional
        Number of bootstrap resampling iterations. Must be >= 1.
        Default is 1000.
    seed : int, optional
        Random seed passed to ``numpy.random.default_rng`` for deterministic
        resampling. Default is 42.

    Returns
    -------
    dict
        A Bootstrap_Result dict with exactly the following keys:

        mean_elite_count : float
            Mean elite count across all bootstrap samples.
        std_elite_count : float
            Standard deviation of elite count across bootstrap samples
            (ddof=1).
        ci_lower_95 : float
            2.5th percentile of the bootstrap distribution.
        ci_upper_95 : float
            97.5th percentile of the bootstrap distribution.
        observed_count : int
            Elite count from the original (non-resampled) DataFrame.

    Raises
    ------
    ValueError
        If ``awi_per_minute`` or ``mean_pqi`` columns are missing from ``df``.
    ValueError
        If ``df`` has fewer than 4 rows.
    ValueError
        If ``n_bootstrap`` is less than 1.

    Notes
    -----
    Determinism is guaranteed by using ``numpy.random.default_rng(seed)``
    with a fixed seed. The same seed always produces the same sequence of
    random integers.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({
    ...     "awi_per_minute": rng.uniform(0, 30, 20),
    ...     "mean_pqi": rng.uniform(0, 100, 20),
    ... })
    >>> result = bootstrap_elite_quadrant(df, n_bootstrap=100, seed=42)
    >>> set(result.keys()) == {
    ...     "mean_elite_count", "std_elite_count",
    ...     "ci_lower_95", "ci_upper_95", "observed_count"
    ... }
    True
    """
    # Validate inputs
    missing = [c for c in ("awi_per_minute", "mean_pqi") if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input DataFrame is missing required column(s): {missing}. "
            "Expected columns: 'awi_per_minute' and 'mean_pqi'."
        )

    if len(df) < 4:
        raise ValueError(
            f"Input DataFrame has {len(df)} row(s), but at least 4 rows are "
            "required to compute a meaningful 75th percentile threshold."
        )

    if n_bootstrap < 1:
        raise ValueError(
            f"n_bootstrap must be >= 1, got {n_bootstrap}."
        )

    # Compute observed count from the original DataFrame
    awi_q75_obs = df["awi_per_minute"].quantile(0.75)
    pqi_q75_obs = df["mean_pqi"].quantile(0.75)
    observed_count = int(
        ((df["awi_per_minute"] >= awi_q75_obs) & (df["mean_pqi"] >= pqi_q75_obs)).sum()
    )

    # Bootstrap resampling
    rng = numpy.random.default_rng(seed)
    counts = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(df), size=len(df))
        sample = df.iloc[idx]
        awi_q75 = sample["awi_per_minute"].quantile(0.75)
        pqi_q75 = sample["mean_pqi"].quantile(0.75)
        count = int(
            ((sample["awi_per_minute"] >= awi_q75) & (sample["mean_pqi"] >= pqi_q75)).sum()
        )
        counts.append(count)

    counts = numpy.array(counts)

    return {
        "mean_elite_count": float(numpy.mean(counts)),
        "std_elite_count":  float(numpy.std(counts, ddof=1)),
        "ci_lower_95":      float(numpy.percentile(counts, 2.5)),
        "ci_upper_95":      float(numpy.percentile(counts, 97.5)),
        "observed_count":   observed_count,
    }
