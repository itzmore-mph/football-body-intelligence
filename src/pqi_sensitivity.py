"""
pqi_sensitivity.py
Grid scan over PQI weight combinations — rank-stability analysis.
No I/O, no S3 access, no side effects.
"""

from dataclasses import dataclass

import pandas as pd
from scipy.stats import spearmanr


@dataclass
class SensitivityResult:
    correlations: pd.DataFrame
    rank_deltas: pd.DataFrame
    baseline_ranking: pd.Series


BASELINE_WEIGHTS: dict[str, float] = {
    "orientation": 0.40,
    "stance": 0.30,
    "proximity": 0.30,
}


def _weight_combo_key(w: dict[str, float]) -> str:
    return f"o{w['orientation']:.2f}_s{w['stance']:.2f}_p{w['proximity']:.2f}"


def generate_weight_grid(step: float = 0.05) -> list[dict[str, float]]:
    """
    Return all weight triplets (w_orientation, w_stance, w_proximity) where each
    value is a multiple of ``step`` and the three values sum to 1.0.

    At step=0.05 this yields C(22, 2) = 231 combinations, including degenerate
    cases where one or two weights are zero.

    Args:
        step: Grid resolution. Must divide 1.0 exactly (e.g. 0.05, 0.10, 0.25).

    Returns:
        List of dicts with keys "orientation", "stance", "proximity".
    """
    n = round(1.0 / step)
    combos: list[dict[str, float]] = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            combos.append({
                "orientation": round(i * step, 10),
                "stance": round(j * step, 10),
                "proximity": round(k * step, 10),
            })
    return combos


def run_sensitivity(
    df: pd.DataFrame,
    weight_grid: list[dict[str, float]],
    player_col: str = "name",
) -> SensitivityResult:
    """
    Run PQI weight sensitivity analysis over all combinations in ``weight_grid``.

    For each weight combination the function re-computes a weighted PQI from the
    per-player mean sub-scores, ranks players (rank 1 = highest PQI), and compares
    that ranking to the baseline (0.40 / 0.30 / 0.30) via Spearman rho and rank delta.

    Args:
        df:          DataFrame with columns ``orientation_mean``, ``stance_mean``,
                     ``proximity_mean``, and ``player_col``. Multiple rows per player
                     (different matches / phases) are aggregated by mean.
        weight_grid: List of weight dicts as returned by ``generate_weight_grid``.
        player_col:  Column used to identify players. Default "name".

    Returns:
        SensitivityResult with:
          correlations   -- one row per weight combo; columns w_orientation, w_stance,
                           w_proximity, spearman_rho.
          rank_deltas    -- indexed by player; one column per weight combo key;
                           value = rank under that combo minus baseline rank
                           (positive = dropped, negative = rose).
          baseline_ranking -- Series indexed by player; values = baseline rank (1 = best).
    """
    player_scores = (
        df.groupby(player_col)[["orientation_mean", "stance_mean", "proximity_mean"]]
        .mean()
    )

    def _rank(w: dict[str, float]) -> pd.Series:
        pqi = (
            w["orientation"] * player_scores["orientation_mean"]
            + w["stance"] * player_scores["stance_mean"]
            + w["proximity"] * player_scores["proximity_mean"]
        )
        return pqi.rank(ascending=False, method="min")

    baseline_ranking = _rank(BASELINE_WEIGHTS)

    correlations: list[dict] = []
    rank_deltas: dict[str, pd.Series] = {}

    for w in weight_grid:
        key = _weight_combo_key(w)
        ranking = _rank(w)
        rho, _ = spearmanr(baseline_ranking.values, ranking.values)
        correlations.append({
            "w_orientation": w["orientation"],
            "w_stance": w["stance"],
            "w_proximity": w["proximity"],
            "spearman_rho": float(rho),
        })
        rank_deltas[key] = ranking - baseline_ranking

    return SensitivityResult(
        correlations=pd.DataFrame(correlations),
        rank_deltas=pd.DataFrame(rank_deltas),
        baseline_ranking=baseline_ranking,
    )


def summary_stats(result: SensitivityResult) -> dict:
    """
    Summarise rank-stability across all weight combinations.

    Args:
        result: SensitivityResult returned by ``run_sensitivity``.

    Returns:
        Dict with keys:
          spearman_min          -- minimum Spearman rho across all weight combos.
          spearman_max          -- maximum Spearman rho (always 1.0 when baseline is in grid).
          spearman_mean         -- mean Spearman rho.
          frac_above_0_9        -- fraction of combos with rho > 0.90.
          most_stable_players   -- list of up to 5 player names with smallest mean |rank_delta|.
          most_volatile_players -- list of up to 5 player names with largest mean |rank_delta|.
    """
    rhos = result.correlations["spearman_rho"]
    mean_abs_delta = result.rank_deltas.abs().mean(axis=1)
    n_top = min(5, len(mean_abs_delta))
    return {
        "spearman_min": float(rhos.min()),
        "spearman_max": float(rhos.max()),
        "spearman_mean": float(rhos.mean()),
        "frac_above_0_9": float((rhos > 0.9).mean()),
        "most_stable_players": mean_abs_delta.nsmallest(n_top).index.tolist(),
        "most_volatile_players": mean_abs_delta.nlargest(n_top).index.tolist(),
    }
