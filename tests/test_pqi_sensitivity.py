import pandas as pd

from src.pqi_sensitivity import SensitivityResult, _weight_combo_key, generate_weight_grid


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
