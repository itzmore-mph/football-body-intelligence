"""
tests/test_benchmark_report.py

Tests for src/benchmark_report.py -- Comparison_Table and BENCHMARK_REFERENCES.
No em dash characters (U+2014) appear anywhere in this file.
"""

import pandas as pd
import pytest

from src.benchmark_report import BENCHMARK_REFERENCES, generate_benchmark_summary


def test_generate_benchmark_summary_returns_dataframe():
    """Validates: Requirement 6.1 -- function returns a pandas DataFrame."""
    result = generate_benchmark_summary()
    assert isinstance(result, pd.DataFrame)


def test_generate_benchmark_summary_columns():
    """Validates: Requirements 5.2, 6.2 -- DataFrame has exactly the required columns."""
    df = generate_benchmark_summary()
    expected_columns = [
        "Sport/Industry",
        "Data Source",
        "Signal Type",
        "Temporal Resolution",
        "Closest AWI/PQI Analog",
        "Key Difference",
    ]
    assert df.columns.tolist() == expected_columns


def test_benchmark_references_min_entries():
    """Validates: Requirements 7.2, 9.3 -- BENCHMARK_REFERENCES has at least 4 entries."""
    assert len(BENCHMARK_REFERENCES) >= 4


def test_generate_benchmark_summary_min_rows():
    """Validates: Requirements 5.3, 6.3 -- DataFrame has at least 4 rows."""
    df = generate_benchmark_summary()
    assert len(df) >= 4


def test_benchmark_references_entry_keys():
    """Validates: Requirement 7.3 -- each entry has sport, author, year, url keys."""
    required_keys = {"sport", "author", "year", "url"}
    for key, entry in BENCHMARK_REFERENCES.items():
        assert required_keys.issubset(entry.keys()), (
            f"Entry '{key}' is missing keys: {required_keys - entry.keys()}"
        )


def test_no_em_dashes_in_module():
    """Validates: Requirement 10.3 -- source file contains no U+2014 em dash characters."""
    source_path = "src/benchmark_report.py"
    with open(source_path, encoding="utf-8") as f:
        source = f.read()
    assert "\u2014" not in source, "Em dash (U+2014) found in src/benchmark_report.py"
