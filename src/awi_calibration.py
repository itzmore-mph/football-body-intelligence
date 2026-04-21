"""
awi_calibration.py

Validates the AWI scan threshold against known reference players by comparing
pre-computed AWI rates from results/awi_full.csv to expected reference values.

No skeleton pipeline is re-run; this module reads the stored CSV output only.
"""

import os
import pandas as pd

# ---------------------------------------------------------------------------
# Reference anchors
# ---------------------------------------------------------------------------

REFERENCE_CASES: dict[str, dict] = {
    "kimmich_fcb_hsv_h1": {
        "player_id":    "Joshua Walter Kimmich",
        "match_label":  "FCB-HSV",
        "phase":        "1st half",
        "expected_awi": 21.77,
    },
    "hojlund_sge_fcb_h1": {
        "player_id":    "Oscar Winther Höjlund",
        "match_label":  "SGE-FCB",
        "phase":        "1st half",
        "expected_awi": 26.90,
    },
}

# Path to the pre-computed AWI results file (relative to the project root).
_AWI_CSV_PATH = os.path.join("results", "awi_full.csv")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_awi_threshold(
    player_id: str,
    match_label: str,
    expected_rate: float,
    tolerance: float = 0.15,
) -> dict:
    """Validate the AWI scan threshold against a known reference player.

    Reads pre-computed AWI values from ``results/awi_full.csv`` and compares
    the stored ``awi_per_minute`` for the given player and match against the
    supplied ``expected_rate``. No skeleton data is re-processed.

    The ``player_id`` is matched exactly against the ``name`` column of the
    CSV. The ``match_label`` is matched exactly against the ``match_id``
    column. When multiple rows match (e.g. different phases), the first
    matching row is used.

    Parameters
    ----------
    player_id : str
        Player identifier matched exactly against the ``name`` column of
        ``results/awi_full.csv`` (e.g. ``"Joshua Walter Kimmich"``).
    match_label : str
        Match identifier matched exactly against the ``match_id`` column of
        the CSV (e.g. ``"FCB-HSV"``).
    expected_rate : float
        Reference AWI rate in scans per minute. Must be strictly positive.
    tolerance : float, optional
        Maximum allowed fractional deviation between the computed and expected
        AWI rates, expressed as a decimal fraction (default ``0.15`` = 15 pct).

    Returns
    -------
    dict
        A ``Validation_Report`` dict with exactly the following keys:

        ``player`` : str
            Player identifier passed as ``player_id``.
        ``match`` : str
            Match label passed as ``match_label``.
        ``phase`` : str
            Phase label taken from the matching CSV row (e.g. ``"1st half"``).
        ``computed_awi`` : float
            AWI rate read from the CSV for the matching row.
        ``expected_awi`` : float
            Reference AWI rate passed as ``expected_rate``.
        ``within_tolerance`` : bool
            ``True`` when ``deviation_pct <= tolerance * 100``.
        ``deviation_pct`` : float
            ``abs(computed_awi - expected_awi) / expected_awi * 100``.

    Raises
    ------
    FileNotFoundError
        If ``results/awi_full.csv`` does not exist.
    ValueError
        If ``expected_rate`` is zero or negative.
    ValueError
        If no rows in the CSV match both ``player_id`` and ``match_label``.

    Examples
    --------
    >>> report = validate_awi_threshold("Joshua Walter Kimmich", "FCB-HSV", 21.77)
    >>> report["within_tolerance"]
    True
    """
    if expected_rate <= 0:
        raise ValueError(
            f"expected_rate must be positive, got {expected_rate!r}."
        )

    if not os.path.exists(_AWI_CSV_PATH):
        raise FileNotFoundError(
            f"AWI results file not found: {_AWI_CSV_PATH!r}. "
            "Run the AWI pipeline to generate it before calling "
            "validate_awi_threshold."
        )

    df = pd.read_csv(_AWI_CSV_PATH)

    mask = (df["name"] == player_id) & (df["match_id"] == match_label)
    matches = df[mask]

    if matches.empty:
        available = (
            df[["name", "match_id"]]
            .drop_duplicates()
            .apply(lambda r: f"{r['name']} / {r['match_id']}", axis=1)
            .tolist()
        )
        available_str = ", ".join(available[:10])
        raise ValueError(
            f"No rows found for player_id={player_id!r} and "
            f"match_label={match_label!r}. "
            f"Available combinations (first 10): {available_str}."
        )

    row = matches.iloc[0]
    computed_awi: float = float(row["awi_per_minute"])
    phase: str = str(row["phase_label"])

    deviation_pct: float = abs(computed_awi - expected_rate) / expected_rate * 100
    within_tolerance: bool = deviation_pct <= tolerance * 100

    return {
        "player":           player_id,
        "match":            match_label,
        "phase":            phase,
        "computed_awi":     computed_awi,
        "expected_awi":     expected_rate,
        "within_tolerance": within_tolerance,
        "deviation_pct":    round(deviation_pct, 4),
    }
