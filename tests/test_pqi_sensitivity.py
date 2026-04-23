import pandas as pd

from src.pqi_sensitivity import (
    SensitivityResult,
    _weight_combo_key,
    generate_weight_grid,
    run_sensitivity,
)


def test_weight_combo_key_baseline():
    w = {"orientation": 0.40, "stance": 0.30, "proximity": 0.30}
    assert _weight_combo_key(w) == "o0.40_s0.30_p0.30"


def test_weight_combo_key_zero():
    w = {"orientation": 1.0, "stance": 0.0, "proximity": 0.0}
    assert _weight_combo_key(w) == "o1.00_s0.00_p0.00"


def test_sensitivity_result_is_dataclass():
    r = SensitivityResult(
        correlations=pd.DataFrame(),
        rank_deltas=pd.DataFrame(),
        baseline_ranking=pd.Series(dtype=float),
    )
    assert hasattr(r, "correlations")
    assert hasattr(r, "rank_deltas")
    assert hasattr(r, "baseline_ranking")


def test_generate_weight_grid_sums_to_one():
    grid = generate_weight_grid(step=0.05)
    for w in grid:
        total = w["orientation"] + w["stance"] + w["proximity"]
        assert abs(total - 1.0) < 1e-9, f"Weights do not sum to 1.0: {w}"


def test_generate_weight_grid_contains_baseline():
    grid = generate_weight_grid(step=0.05)
    keys = [_weight_combo_key(w) for w in grid]
    assert "o0.40_s0.30_p0.30" in keys


def test_generate_weight_grid_count_default():
    # step=0.05 → n=20; non-negative integer triplets summing to 20 = C(22,2) = 231
    grid = generate_weight_grid(step=0.05)
    assert len(grid) == 231


def test_generate_weight_grid_count_coarse():
    # step=0.25 → n=4; C(6,2) = 15
    grid = generate_weight_grid(step=0.25)
    assert len(grid) == 15


def test_generate_weight_grid_all_nonnegative():
    grid = generate_weight_grid(step=0.05)
    for w in grid:
        assert w["orientation"] >= 0.0
        assert w["stance"] >= 0.0
        assert w["proximity"] >= 0.0


def _make_player_df() -> pd.DataFrame:
    """Five players, one row each (single match + phase)."""
    return pd.DataFrame({
        "name":             ["Alice", "Bob", "Carol", "Dave", "Eve"],
        "orientation_mean": [80.0,   60.0,  70.0,   50.0,   40.0],
        "stance_mean":      [30.0,   70.0,  50.0,   60.0,   80.0],
        "proximity_mean":   [90.0,   40.0,  60.0,   70.0,   50.0],
        "match_id":         ["M1"] * 5,
        "phase_label":      ["1st half"] * 5,
    })


def test_run_sensitivity_baseline_rho_is_one():
    df = _make_player_df()
    # step=0.05 grid includes the baseline (0.40, 0.30, 0.30)
    grid_fine = generate_weight_grid(step=0.05)
    result_fine = run_sensitivity(df, grid_fine)
    row_fine = result_fine.correlations[
        (result_fine.correlations["w_orientation"] == 0.40) &
        (result_fine.correlations["w_stance"] == 0.30) &
        (result_fine.correlations["w_proximity"] == 0.30)
    ]
    assert len(row_fine) == 1
    assert abs(float(row_fine["spearman_rho"].iloc[0]) - 1.0) < 1e-9


def test_run_sensitivity_rank_deltas_shape():
    df = _make_player_df()
    grid = generate_weight_grid(step=0.25)
    result = run_sensitivity(df, grid)
    n_players = df["name"].nunique()
    n_combos = len(grid)
    assert result.rank_deltas.shape == (n_players, n_combos)


def test_run_sensitivity_rank_deltas_baseline_is_zero():
    df = _make_player_df()
    grid = generate_weight_grid(step=0.05)
    result = run_sensitivity(df, grid)
    baseline_key = _weight_combo_key({"orientation": 0.40, "stance": 0.30, "proximity": 0.30})
    assert (result.rank_deltas[baseline_key] == 0).all()


def test_run_sensitivity_correlations_columns():
    df = _make_player_df()
    grid = generate_weight_grid(step=0.25)
    result = run_sensitivity(df, grid)
    assert set(result.correlations.columns) == {
        "w_orientation", "w_stance", "w_proximity", "spearman_rho"
    }


def test_run_sensitivity_rho_in_range():
    df = _make_player_df()
    grid = generate_weight_grid(step=0.25)
    result = run_sensitivity(df, grid)
    assert result.correlations["spearman_rho"].between(-1.0, 1.0).all()
