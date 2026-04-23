import pandas as pd

from src.pqi_sensitivity import SensitivityResult, _weight_combo_key


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
