"""
tests/test_benchmark_reference.py

Unit tests for src/benchmark_reference.py

Covers:
  - Catalogue completeness and structural integrity
  - percentile_in_reference: boundary values, monotonicity, cohort switching
  - build_comparison_table: None filtering, value mapping, output shape
  - sample_reference_distribution: shape, bounds, reproducibility
"""

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

# ---------------------------------------------------------------------------
# Catalogue integrity
# ---------------------------------------------------------------------------


class TestCatalogue:

    def test_catalogue_not_empty(self):
        assert len(REFERENCE_CATALOGUE) >= 5

    def test_all_metric_types_present(self):
        types = {r.metric_type for r in REFERENCE_CATALOGUE}
        assert "AWI" in types
        assert "PQI" in types
        assert "PQI_orientation" in types
        assert "PQI_stance" in types
        assert "PQI_proximity" in types

    def test_all_fields_non_empty(self):
        for ref in REFERENCE_CATALOGUE:
            assert ref.system, f"Empty system in {ref}"
            assert ref.sport, f"Empty sport in {ref}"
            assert ref.metric_name, f"Empty metric_name in {ref}"
            assert ref.unit, f"Empty unit in {ref}"
            assert ref.source, f"Empty source in {ref}"
            assert ref.concept, f"Empty concept in {ref}"

    def test_means_positive(self):
        for ref in REFERENCE_CATALOGUE:
            assert ref.mean > 0, f"Non-positive mean in {ref.system}"
            assert ref.elite_mean > 0, f"Non-positive elite_mean in {ref.system}"

    def test_stds_positive(self):
        for ref in REFERENCE_CATALOGUE:
            assert ref.std > 0, f"Non-positive std in {ref.system}"
            assert ref.elite_std > 0, f"Non-positive elite_std in {ref.system}"

    def test_elite_mean_gte_population_mean(self):
        """Elite cohort should score at least as high as the population."""
        for ref in REFERENCE_CATALOGUE:
            assert ref.elite_mean >= ref.mean, (
                f"{ref.system}: elite_mean ({ref.elite_mean}) < mean ({ref.mean})"
            )

    def test_score_metrics_in_range(self):
        """Non-AWI metrics should have means in [0, 100]."""
        for ref in REFERENCE_CATALOGUE:
            if ref.metric_type != "AWI":
                assert 0 <= ref.mean <= 100, f"{ref.system} mean out of [0,100]"
                assert 0 <= ref.elite_mean <= 100, f"{ref.system} elite_mean out of [0,100]"

    def test_get_all_references_returns_full_list(self):
        assert get_all_references() == REFERENCE_CATALOGUE

    def test_get_references_for_metric_awi(self):
        refs = get_references_for_metric("AWI")
        assert len(refs) >= 1
        assert all(r.metric_type == "AWI" for r in refs)

    def test_get_references_for_metric_unknown(self):
        refs = get_references_for_metric("NONEXISTENT")
        assert refs == []

    def test_no_em_dashes_in_text_fields(self):
        """No em dashes allowed in any generated text (per tech.md rules)."""
        for ref in REFERENCE_CATALOGUE:
            for field_val in (ref.concept, ref.source, ref.metric_name, ref.system):
                assert "\u2014" not in field_val, (
                    f"Em dash found in {ref.system}: {field_val!r}"
                )


# ---------------------------------------------------------------------------
# percentile_in_reference
# ---------------------------------------------------------------------------


class TestPercentileInReference:

    @pytest.fixture
    def ref(self):
        return ReferenceDistribution(
            system="Test", sport="Test", metric_name="Test",
            metric_type="PQI", mean=50.0, std=10.0,
            elite_mean=70.0, elite_std=8.0,
            unit="score", source="test", concept="test",
        )

    def test_at_mean_is_50th_percentile(self, ref):
        pct = percentile_in_reference(50.0, ref, "population")
        assert pct == pytest.approx(50.0, abs=0.5)

    def test_above_mean_above_50(self, ref):
        pct = percentile_in_reference(60.0, ref, "population")
        assert pct > 50.0

    def test_below_mean_below_50(self, ref):
        pct = percentile_in_reference(40.0, ref, "population")
        assert pct < 50.0

    def test_output_in_0_100(self, ref):
        for val in (-100.0, 0.0, 50.0, 100.0, 200.0):
            pct = percentile_in_reference(val, ref, "population")
            assert 0.0 <= pct <= 100.0

    def test_monotone_increasing(self, ref):
        values = np.linspace(0, 100, 50)
        pcts = [percentile_in_reference(v, ref, "population") for v in values]
        assert all(pcts[i] <= pcts[i + 1] for i in range(len(pcts) - 1))

    def test_elite_cohort_uses_elite_params(self, ref):
        # At elite_mean (70), elite percentile should be ~50
        pct_elite = percentile_in_reference(70.0, ref, "elite")
        assert pct_elite == pytest.approx(50.0, abs=1.0)

    def test_population_vs_elite_differ(self, ref):
        # Same value, different cohort -> different percentile
        pct_pop = percentile_in_reference(70.0, ref, "population")
        pct_elite = percentile_in_reference(70.0, ref, "elite")
        assert pct_pop != pytest.approx(pct_elite, abs=1.0)

    def test_zero_std_returns_boundary(self):
        ref_zero = ReferenceDistribution(
            system="T", sport="T", metric_name="T", metric_type="PQI",
            mean=50.0, std=0.0, elite_mean=50.0, elite_std=0.0,
            unit="score", source="t", concept="t",
        )
        assert percentile_in_reference(60.0, ref_zero) == 100.0
        assert percentile_in_reference(40.0, ref_zero) == 0.0


# ---------------------------------------------------------------------------
# build_comparison_table
# ---------------------------------------------------------------------------


class TestBuildComparisonTable:

    def test_returns_list_of_dicts(self):
        rows = build_comparison_table(awi_value=25.0, pqi_value=65.0)
        assert isinstance(rows, list)
        assert all(isinstance(r, dict) for r in rows)

    def test_required_keys_present(self):
        rows = build_comparison_table(awi_value=25.0, pqi_value=65.0)
        required = {
            "system", "sport", "metric_name", "metric_type",
            "player_value", "ref_mean", "ref_elite_mean",
            "pct_vs_population", "pct_vs_elite", "unit", "concept", "source",
        }
        for row in rows:
            assert required.issubset(row.keys()), f"Missing keys in {row}"

    def test_none_values_excluded(self):
        rows_all = build_comparison_table(
            awi_value=25.0, pqi_value=65.0,
            orientation_value=70.0, stance_value=60.0, proximity_value=55.0,
        )
        rows_partial = build_comparison_table(awi_value=25.0, pqi_value=None)
        # Partial should have fewer rows
        assert len(rows_partial) < len(rows_all)

    def test_nan_values_excluded(self):
        rows = build_comparison_table(awi_value=float("nan"), pqi_value=65.0)
        types = [r["metric_type"] for r in rows]
        assert "AWI" not in types

    def test_all_none_returns_empty(self):
        rows = build_comparison_table(
            awi_value=None, pqi_value=None,
            orientation_value=None, stance_value=None, proximity_value=None,
        )
        assert rows == []

    def test_player_value_matches_input(self):
        rows = build_comparison_table(awi_value=27.5, pqi_value=None)
        awi_rows = [r for r in rows if r["metric_type"] == "AWI"]
        assert len(awi_rows) >= 1
        assert awi_rows[0]["player_value"] == pytest.approx(27.5, abs=0.01)

    def test_percentiles_in_range(self):
        rows = build_comparison_table(
            awi_value=25.0, pqi_value=65.0,
            orientation_value=70.0, stance_value=60.0, proximity_value=55.0,
        )
        for row in rows:
            assert 0.0 <= row["pct_vs_population"] <= 100.0
            assert 0.0 <= row["pct_vs_elite"] <= 100.0

    def test_high_value_high_percentile(self):
        rows = build_comparison_table(pqi_value=99.0, awi_value=None)
        for row in rows:
            if row["metric_type"] == "PQI":
                assert row["pct_vs_population"] > 90.0

    def test_low_value_low_percentile(self):
        rows = build_comparison_table(pqi_value=1.0, awi_value=None)
        for row in rows:
            if row["metric_type"] == "PQI":
                assert row["pct_vs_population"] < 10.0


# ---------------------------------------------------------------------------
# sample_reference_distribution
# ---------------------------------------------------------------------------


class TestSampleReferenceDistribution:

    @pytest.fixture
    def awi_ref(self):
        return next(r for r in REFERENCE_CATALOGUE if r.metric_type == "AWI")

    @pytest.fixture
    def pqi_ref(self):
        return next(r for r in REFERENCE_CATALOGUE if r.metric_type == "PQI")

    def test_returns_correct_shape(self, pqi_ref):
        samples = sample_reference_distribution(pqi_ref, n=200)
        assert samples.shape == (200,)

    def test_score_samples_in_0_100(self, pqi_ref):
        samples = sample_reference_distribution(pqi_ref, n=500)
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 100.0)

    def test_awi_samples_in_0_60(self, awi_ref):
        samples = sample_reference_distribution(awi_ref, n=500)
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 60.0)

    def test_reproducible_with_same_seed(self, pqi_ref):
        s1 = sample_reference_distribution(pqi_ref, n=100, rng_seed=7)
        s2 = sample_reference_distribution(pqi_ref, n=100, rng_seed=7)
        np.testing.assert_array_equal(s1, s2)

    def test_different_seeds_differ(self, pqi_ref):
        s1 = sample_reference_distribution(pqi_ref, n=100, rng_seed=1)
        s2 = sample_reference_distribution(pqi_ref, n=100, rng_seed=2)
        assert not np.array_equal(s1, s2)

    def test_elite_cohort_higher_mean(self, pqi_ref):
        pop = sample_reference_distribution(pqi_ref, n=2000, cohort="population")
        elite = sample_reference_distribution(pqi_ref, n=2000, cohort="elite")
        assert elite.mean() > pop.mean()
