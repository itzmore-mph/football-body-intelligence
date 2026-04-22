# tests/test_benchmark.py
import math
import numpy as np
import pytest

from src.benchmark_reference import (
    REFERENCE_CATALOGUE,
    ReferenceDistribution,
    build_comparison_table,
    get_all_references,
    get_references_for_metric,
    percentile_in_reference,
    sample_reference_distribution,
)


class TestGetAllReferences:
    def test_returns_six_entries(self):
        refs = get_all_references()
        assert len(refs) == 6

    def test_all_entries_have_positive_parameters(self):
        for ref in get_all_references():
            assert ref.mean > 0
            assert ref.std > 0
            assert ref.elite_mean > 0

    def test_all_entries_are_reference_distribution(self):
        for ref in get_all_references():
            assert isinstance(ref, ReferenceDistribution)


class TestGetReferencesForMetric:
    def test_awi_returns_one_entry(self):
        refs = get_references_for_metric("AWI")
        assert len(refs) == 1

    def test_awi_unit_is_scans_per_min(self):
        ref = get_references_for_metric("AWI")[0]
        assert ref.unit == "scans/min"

    def test_unknown_metric_returns_empty_list(self):
        assert get_references_for_metric("unknown_metric") == []

    def test_pqi_stance_returns_two_entries(self):
        assert len(get_references_for_metric("PQI_stance")) == 2


class TestPercentileInReference:
    def test_mean_value_returns_near_50(self):
        ref = get_references_for_metric("AWI")[0]
        pct = percentile_in_reference(ref.mean, ref)
        assert 49.0 <= pct <= 51.0

    def test_elite_mean_returns_above_75(self):
        ref = get_references_for_metric("AWI")[0]
        assert percentile_in_reference(ref.elite_mean, ref) > 75.0

    def test_nan_raises_value_error(self):
        ref = get_references_for_metric("AWI")[0]
        with pytest.raises(ValueError):
            percentile_in_reference(float("nan"), ref)

    def test_inf_raises_value_error(self):
        ref = get_references_for_metric("AWI")[0]
        with pytest.raises(ValueError):
            percentile_in_reference(float("inf"), ref)


class TestSampleReferenceDistribution:
    def test_population_returns_correct_length(self):
        ref = get_references_for_metric("AWI")[0]
        assert len(sample_reference_distribution(ref, n=100)) == 100

    def test_population_samples_near_mean(self):
        ref = get_references_for_metric("AWI")[0]
        samples = sample_reference_distribution(ref, n=2000)
        assert abs(samples.mean() - ref.mean) < 1.5

    def test_elite_mean_higher_than_population(self):
        ref = get_references_for_metric("AWI")[0]
        elite = sample_reference_distribution(ref, n=500, cohort="elite")
        pop = sample_reference_distribution(ref, n=500, cohort="population")
        assert elite.mean() > pop.mean()

    def test_invalid_cohort_raises_value_error(self):
        ref = get_references_for_metric("AWI")[0]
        with pytest.raises(ValueError):
            sample_reference_distribution(ref, n=10, cohort="invalid")

    def test_n_zero_raises_value_error(self):
        ref = get_references_for_metric("AWI")[0]
        with pytest.raises(ValueError):
            sample_reference_distribution(ref, n=0)


class TestBuildComparisonTable:
    def test_all_none_returns_empty(self):
        assert build_comparison_table() == []

    def test_awi_value_returns_one_row(self):
        rows = build_comparison_table(awi_value=24.0)
        assert len(rows) == 1
        assert rows[0]["metric_type"] == "AWI"

    def test_row_has_required_keys(self):
        rows = build_comparison_table(awi_value=24.0)
        required = {"system", "sport", "metric_type", "pct_vs_population", "pct_vs_elite", "unit"}
        assert required.issubset(rows[0].keys())

    def test_pct_vs_population_in_range(self):
        rows = build_comparison_table(
            awi_value=24.0, pqi_value=60.0,
            orientation_value=62.0, stance_value=58.0, proximity_value=55.0,
        )
        for row in rows:
            assert 0.0 <= row["pct_vs_population"] <= 100.0

    def test_none_values_excluded(self):
        rows = build_comparison_table(awi_value=24.0, pqi_value=None, orientation_value=None)
        assert all(r["metric_type"] == "AWI" for r in rows)


from src.benchmark_report import BENCHMARK_REFERENCES, generate_benchmark_summary
import pandas as pd


class TestBenchmarkReferences:
    def test_has_four_keys(self):
        assert len(BENCHMARK_REFERENCES) == 4

    def test_expected_keys_present(self):
        expected = {
            "nba_second_spectrum",
            "nfl_next_gen_stats",
            "cricket_hawk_eye",
            "industrial_motion_capture",
        }
        assert set(BENCHMARK_REFERENCES.keys()) == expected

    def test_each_entry_has_required_fields(self):
        required = {"system", "sport", "metric_analog", "citation"}
        for key, entry in BENCHMARK_REFERENCES.items():
            missing = required - set(entry.keys())
            assert not missing, f"{key} missing fields: {missing}"

    def test_all_string_values_non_empty(self):
        for key, entry in BENCHMARK_REFERENCES.items():
            for field, value in entry.items():
                assert isinstance(value, str) and len(value) > 0, (
                    f"{key}.{field} is empty"
                )


class TestGenerateBenchmarkSummary:
    def test_returns_six_rows(self):
        df = generate_benchmark_summary()
        assert len(df) == 6

    def test_has_required_columns(self):
        df = generate_benchmark_summary()
        for col in ["System", "Sport", "Maps to", "Key distinction", "Citation"]:
            assert col in df.columns

    def test_no_null_values(self):
        df = generate_benchmark_summary()
        assert not df.isnull().any().any()

    def test_all_values_non_empty_strings(self):
        df = generate_benchmark_summary()
        for col in df.columns:
            assert (df[col].str.len() > 0).all(), f"Column {col} has empty strings"
