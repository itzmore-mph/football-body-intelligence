"""
Position-adjusted PQI normalisation module.

Computes z-scores of ``pqi_mean`` within four position groups (GK, DEF, MID, FWD),
removing the structural bias that arises from comparing goalkeepers and outfield
players on a single raw scale.
"""

import pandas as pd

POSITION_GROUPS: dict[str, str] = {
    # Goalkeepers
    "TW":  "GK",
    # Defenders
    "IVL": "DEF", "IVR": "DEF", "IVZ": "DEF",
    "LA":  "DEF", "RA":  "DEF", "LV":  "DEF", "RV":  "DEF",
    # Midfielders
    "DML": "MID", "DMR": "MID", "DMZ": "MID", "DLM": "MID", "DRM": "MID",
    "ZO":  "MID", "RM":  "MID",
    "OHL": "MID", "OHR": "MID", "OLM": "MID", "ORM": "MID",
    "HL":  "MID", "HR":  "MID",
    # Forwards
    "STL": "FWD", "STR": "FWD", "STZ": "FWD",
}


def normalize_pqi_by_position(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add position-adjusted PQI z-scores to a player DataFrame.

    Each player's ``pqi_mean`` is standardised within their position group
    (GK, DEF, MID, or FWD) using the z-score formula::

        pqi_position_adjusted = (pqi_mean - group_mean) / group_std

    where ``group_mean`` and ``group_std`` are computed from all players in the
    same position group present in ``df``.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least the columns ``player_id``,
        ``position_code``, and ``pqi_mean``.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with two additional columns appended:

        ``pqi_position_adjusted`` : float
            Z-score of ``pqi_mean`` within the player's position group.
            Set to NaN when the position group contains fewer than 2 players
            or when ``position_code`` is not present in ``POSITION_GROUPS``.

        ``position_adjusted`` : bool
            True for every row.

    Raises
    ------
    ValueError
        If any of the required columns (``player_id``, ``position_code``,
        ``pqi_mean``) are absent from ``df``. The error message lists all
        missing column names.

    Notes
    -----
    Unknown ``position_code`` values are silently mapped to NaN and receive
    NaN for ``pqi_position_adjusted``. This is not treated as an error.

    Groups with fewer than 2 players produce NaN z-scores because
    ``pandas.Series.std`` with ``ddof=1`` returns NaN for a single-element
    series.

    No existing column in ``df`` is modified.
    """
    required_columns = {"player_id", "position_code", "pqi_mean"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Input DataFrame is missing required columns: {sorted(missing)}"
        )

    # Map position codes to coarse groups; unknown codes become NaN.
    pos_group = df["position_code"].map(POSITION_GROUPS)

    # Compute group mean and std aligned to the original index via transform.
    group_mean = df["pqi_mean"].groupby(pos_group).transform("mean")
    group_std = df["pqi_mean"].groupby(pos_group).transform("std")

    result = df.copy()
    result["pqi_position_adjusted"] = (df["pqi_mean"] - group_mean) / group_std
    result["position_adjusted"] = True

    return result
