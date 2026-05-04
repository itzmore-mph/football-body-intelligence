import json
import re
import src.eda_helpers  # noqa: F401
from unittest.mock import MagicMock, patch
from hypothesis import given, settings
from hypothesis import strategies as st
import pyarrow as pa

from src.eda_helpers import list_bucket, load_json, sample_parquet


# Strategy for generating S3 object dicts with 'Key' and 'Size'
def s3_object_strategy():
    return st.fixed_dictionaries({
        "Key": st.text(min_size=1),
        "Size": st.integers(min_value=0),
    })


# Feature: football-3d-eda-notebook, Property 1: Complete S3 listing across pages
# Validates: Requirements 1.1, 1.2
@given(pages=st.lists(st.lists(s3_object_strategy(), min_size=0), min_size=1))
@settings(max_examples=100)
def test_list_bucket_completeness(pages):
    """Property 1: list_bucket returns all objects across all pages."""
    # Build a mock paginator that yields pages with "Contents"
    mock_pages = [{"Contents": page} if page else {} for page in pages]

    mock_paginator = MagicMock()
    mock_paginator.paginate.return_value = mock_pages

    mock_client = MagicMock()
    mock_client.get_paginator.return_value = mock_paginator

    result = list_bucket(mock_client, "test-bucket")

    # Total count must equal sum of all objects across all pages
    expected_count = sum(len(p) for p in pages)
    assert len(result) == expected_count

    # Every key from every page must appear in the result
    all_keys = [obj["Key"] for page in pages for obj in page]
    result_keys = [obj["Key"] for obj in result]
    assert sorted(result_keys) == sorted(all_keys)


# Feature: football-3d-eda-notebook, Property 3: JSON round-trip
# Validates: Requirements 3.2, 3.3
@given(data=st.dictionaries(st.text(), st.one_of(st.integers(), st.text(), st.booleans())))
@settings(max_examples=100)
def test_json_roundtrip(data):
    """Property 3: load_json returns a dict containing all top-level keys from the original data."""
    raw_bytes = json.dumps(data).encode()

    mock_body = MagicMock()
    mock_body.read.return_value = raw_bytes

    mock_client = MagicMock()
    mock_client.get_object.return_value = {"Body": mock_body}

    result = load_json(mock_client, "test-bucket", "test-key")

    assert isinstance(result, dict)
    for key in data:
        assert key in result


# Strategy for generating pyarrow Tables with at least 1 row
def arrow_table_strategy():
    return st.builds(
        lambda nrows, col_names: pa.table(
            {name: pa.array(list(range(nrows)), type=pa.int64()) for name in col_names},
        ),
        nrows=st.integers(min_value=1, max_value=100),
        col_names=st.lists(
            st.text(min_size=1, max_size=10),
            min_size=1,
            max_size=5,
            unique=True,
        ),
    )


# Feature: football-3d-eda-notebook, Property 5: Parquet sample row count
# Validates: Requirements 5.3
@given(table=arrow_table_strategy())
@settings(max_examples=100)
def test_parquet_to_pandas_row_count(table):
    """Property 5: sample_parquet returns a DataFrame with the same row and column count as the source Table."""
    mock_batch = table.to_batches()[0]

    mock_pf = MagicMock()
    mock_pf.iter_batches.return_value = iter([mock_batch])

    with patch("src.eda_helpers.pq.ParquetFile", return_value=mock_pf):
        result = sample_parquet(fs=MagicMock(), path="fake/path.parquet")

    assert result is not None
    assert result.shape[0] == table.num_rows
    assert result.shape[1] == table.num_columns


import io
import sys
import xml.etree.ElementTree as ET
import pandas as pd
import pyarrow as pa

from src.eda_helpers import report_dataframe, report_dict, report_xml


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def dataframe_strategy():
    """Generate pandas DataFrames with 1–20 rows and 1–5 int64 columns."""
    return st.builds(
        lambda nrows, col_names: pd.DataFrame(
            {name: list(range(nrows)) for name in col_names},
        ),
        nrows=st.integers(min_value=1, max_value=20),
        col_names=st.lists(
            st.text(min_size=1, max_size=10),
            min_size=1,
            max_size=5,
            unique=True,
        ),
    )


def arrow_schema_strategy():
    """Generate pyarrow schemas with 1–5 int64 fields."""
    return st.builds(
        lambda field_names: pa.schema(
            [pa.field(name, pa.int64()) for name in field_names]
        ),
        field_names=st.lists(
            st.text(min_size=1, max_size=10),
            min_size=1,
            max_size=5,
            unique=True,
        ),
    )


def xml_tree_strategy():
    """Generate ET.Element trees with a root element and 0–5 child elements."""
    tag_text = st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll"), min_codepoint=65),
        min_size=1,
        max_size=10,
    )
    attr_strategy = st.dictionaries(
        tag_text,
        st.text(
            alphabet=st.characters(whitelist_categories=("Lu", "Ll"), min_codepoint=65),
            min_size=1,
            max_size=10,
        ),
        max_size=3,
    )
    child_strategy = st.builds(
        lambda tag: ET.Element(tag),
        tag=tag_text,
    )
    return st.builds(
        lambda root_tag, attrs, children: _build_xml_tree(root_tag, attrs, children),
        root_tag=tag_text,
        attrs=attr_strategy,
        children=st.lists(child_strategy, min_size=0, max_size=5),
    )


def _build_xml_tree(root_tag, attrs, children):
    root = ET.Element(root_tag, attrib=attrs)
    for child in children:
        root.append(child)
    return root


# ---------------------------------------------------------------------------
# Property 6: DataFrame report completeness
# Validates: Requirements 6.1, 6.2, 6.3
# ---------------------------------------------------------------------------

@given(df=dataframe_strategy(), name=st.text(min_size=1))
@settings(max_examples=100)
def test_dataframe_report_completeness(df, name):
    """Property 6: report_dataframe output contains name header, shape, all dtype names, and first 5 rows."""
    captured = io.StringIO()
    sys.stdout = captured
    try:
        report_dataframe(name, df)
    finally:
        sys.stdout = sys.__stdout__

    output = captured.getvalue()

    # Name header must appear
    assert name in output

    # Shape tuple must appear
    assert str(df.shape) in output

    # All dtype names must appear
    for dtype in df.dtypes:
        assert str(dtype) in output

    # First 5 rows must appear — verify the head representation is present
    head_str = str(df.head())
    assert head_str in output


# ---------------------------------------------------------------------------
# Property 7: Parquet schema report completeness
# Validates: Requirements 6.4
# ---------------------------------------------------------------------------

@given(schema=arrow_schema_strategy())
@settings(max_examples=100)
def test_schema_report_completeness(schema):
    """Property 7: report_dataframe with schema contains every field name and type string."""
    # Build a minimal DataFrame matching the schema
    df = pd.DataFrame(
        {field.name: pd.array([], dtype="int64") for field in schema}
    )

    captured = io.StringIO()
    sys.stdout = captured
    try:
        report_dataframe("Schema Test", df, schema=schema)
    finally:
        sys.stdout = sys.__stdout__

    output = captured.getvalue()

    for field in schema:
        assert field.name in output
        assert str(field.type) in output


# ---------------------------------------------------------------------------
# Property 8: Metadata key/type report completeness
# Validates: Requirements 6.5
# ---------------------------------------------------------------------------

@given(d=st.dictionaries(st.text(), st.one_of(st.integers(), st.text(), st.none())))
@settings(max_examples=100)
def test_dict_report_completeness(d):
    """Property 8: report_dict output contains every key and type(v).__name__ for each value."""
    captured = io.StringIO()
    sys.stdout = captured
    try:
        report_dict("Dict Test", d)
    finally:
        sys.stdout = sys.__stdout__

    output = captured.getvalue()

    for k, v in d.items():
        assert k in output
        assert type(v).__name__ in output


# ---------------------------------------------------------------------------
# Property 4: XML inspection output completeness
# Validates: Requirements 4.2, 4.3, 4.4
# ---------------------------------------------------------------------------

@given(xml_tree=xml_tree_strategy())
@settings(max_examples=100)
def test_xml_report_completeness(xml_tree):
    """Property 4: report_xml output contains root tag, all child tags, child count, and root attributes."""
    captured = io.StringIO()
    sys.stdout = captured
    try:
        report_xml("XML Test", xml_tree)
    finally:
        sys.stdout = sys.__stdout__

    output = captured.getvalue()

    # Root tag must appear
    assert xml_tree.tag in output

    # All child tags must appear
    children = list(xml_tree)
    for child in children:
        assert child.tag in output

    # Child count must appear
    assert str(len(children)) in output

    # Root attributes must appear
    for attr_key in xml_tree.attrib:
        assert attr_key in output


# ---------------------------------------------------------------------------
# Property 2: No hardcoded AWS credentials
# Validates: Requirements 2.2
# ---------------------------------------------------------------------------

AWS_ACCESS_KEY_PATTERN = re.compile(r"AKIA[0-9A-Z]{16}")
AWS_SECRET_KEY_PATTERN = re.compile(r"[A-Za-z0-9/+=]{40}")


@given(notebook_cells=st.lists(st.text()))
@settings(max_examples=100)
def test_no_hardcoded_credentials(notebook_cells):
    """Property 2: No generated text cell should match AWS credential patterns."""
    for cell in notebook_cells:
        assert not AWS_ACCESS_KEY_PATTERN.search(cell), (
            f"Cell contains AWS Access Key ID pattern: {cell!r}"
        )
        # Only flag 40-char alphanumeric strings that are standalone tokens
        # (i.e. the entire cell or a whitespace-delimited token is exactly 40 chars)
        tokens = cell.split()
        for token in tokens:
            if len(token) == 40 and AWS_SECRET_KEY_PATTERN.fullmatch(token):
                raise AssertionError(
                    f"Cell contains a 40-char alphanumeric token matching AWS secret key pattern: {token!r}"
                )


def test_no_hardcoded_credentials_in_notebook():
    """Notebooks must not contain hardcoded AWS credentials."""
    import json as _json
    import pathlib
    import glob

    notebooks = sorted(glob.glob("notebooks/*.ipynb"))
    assert notebooks, "No notebooks found in notebooks/"

    for nb_path in notebooks:
        notebook = _json.loads(pathlib.Path(nb_path).read_text())
        source_cells = []
        for cell in notebook.get("cells", []):
            source = cell.get("source", [])
            if isinstance(source, list):
                source_cells.append("".join(source))
            elif isinstance(source, str):
                source_cells.append(source)

        for cell_text in source_cells:
            assert not AWS_ACCESS_KEY_PATTERN.search(cell_text), (
                f"{nb_path}: cell contains AWS Access Key ID pattern: {cell_text[:80]!r}"
            )
            tokens = cell_text.split()
            for token in tokens:
                if len(token) == 40 and AWS_SECRET_KEY_PATTERN.fullmatch(token):
                    raise AssertionError(
                        f"{nb_path}: cell contains a 40-char token matching AWS secret key pattern: {token!r}"
                    )


# ===========================================================================
# Feature: football-body-intelligence-platform — PQI Sub-Score Properties
# ===========================================================================

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from src.pqi_calculator import (
    compute_orientation_score,
    compute_stance_score,
    compute_proximity_score,
    compute_pqi,
    compute_knee_flexion,
)

_angle = st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False)
_score = st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)
_distance = st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
_knee_flex = st.floats(min_value=0.0, max_value=180.0, allow_nan=False, allow_infinity=False)
_coord = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)


# ---------------------------------------------------------------------------
# Property 1: Orientation Score Bounds
# Validates: Requirements 1.1, 1.4
# ---------------------------------------------------------------------------

@given(body_yaw=_angle, ball_dir=_angle)
@settings(max_examples=200)
def test_orientation_score_bounds(body_yaw, ball_dir):
    """Property 1: orientation score is always in [0, 100] for any angles in [-180, 180]."""
    result = compute_orientation_score(
        np.array([body_yaw]), np.array([ball_dir])
    )
    score = float(result[0])
    assert 0.0 <= score <= 100.0, f"score={score} out of [0, 100] for yaw={body_yaw}, dir={ball_dir}"


# ---------------------------------------------------------------------------
# Property 2: Orientation Score Symmetry
# Validates: Requirements 1.1, 1.5
# ---------------------------------------------------------------------------

@given(a=_angle, b=_angle)
@settings(max_examples=200)
def test_orientation_score_symmetry(a, b):
    """Property 2: compute_orientation_score(a, b) == compute_orientation_score(b, a)."""
    score_ab = float(compute_orientation_score(np.array([a]), np.array([b]))[0])
    score_ba = float(compute_orientation_score(np.array([b]), np.array([a]))[0])
    assert abs(score_ab - score_ba) < 1e-9, (
        f"Symmetry violated: score({a},{b})={score_ab} != score({b},{a})={score_ba}"
    )


# ---------------------------------------------------------------------------
# Property 3: Stance Score Bounds and Peak
# Validates: Requirements 2.1, 2.3
# ---------------------------------------------------------------------------

@given(knee_flex=_knee_flex)
@settings(max_examples=200)
def test_stance_score_bounds(knee_flex):
    """Property 3a: stance score is always in [0, 100] for knee flexion in [0, 180]."""
    result = compute_stance_score(np.array([knee_flex]))
    score = float(result[0])
    assert 0.0 <= score <= 100.0, f"score={score} out of [0, 100] for knee_flex={knee_flex}"


@given(knee_flex=_knee_flex)
@settings(max_examples=200)
def test_stance_score_peak_at_130(knee_flex):
    """Property 3b: stance score at 130° is >= stance score at any other angle in [0, 180]."""
    peak_score = float(compute_stance_score(np.array([130.0]))[0])
    other_score = float(compute_stance_score(np.array([knee_flex]))[0])
    assert peak_score >= other_score - 1e-9, (
        f"Peak at 130° ({peak_score}) < score at {knee_flex}° ({other_score})"
    )


# ---------------------------------------------------------------------------
# Property 4: Knee Flexion Bounds
# Validates: Requirements 2.4, 2.5
# ---------------------------------------------------------------------------

@given(
    kx=_coord, ky=_coord,
    hx=_coord, hy=_coord,
    ax=_coord, ay=_coord,
)
@settings(max_examples=200)
def test_knee_flexion_bounds_non_degenerate(kx, ky, hx, hy, ax, ay):
    """Property 4a: knee flexion is in [0, 180] or NaN; never raises an exception."""
    result = compute_knee_flexion(
        np.array([kx]), np.array([ky]),
        np.array([hx]), np.array([hy]),
        np.array([ax]), np.array([ay]),
    )
    val = float(result[0])
    if not np.isnan(val):
        assert 0.0 <= val <= 180.0 + 1e-9, f"knee_flexion={val} out of [0, 180]"


def test_knee_flexion_degenerate_zero_vector():
    """Property 4b: zero-length vectors produce NaN (not an exception)."""
    # knee == hip → zero-length v1
    result = compute_knee_flexion(
        np.array([0.0]), np.array([0.0]),
        np.array([0.0]), np.array([0.0]),  # knee == hip
        np.array([1.0]), np.array([0.0]),
    )
    assert np.isnan(result[0]), "Expected NaN for zero-length vector"

    # knee == ankle → zero-length v2
    result2 = compute_knee_flexion(
        np.array([0.0]), np.array([0.0]),
        np.array([1.0]), np.array([0.0]),
        np.array([0.0]), np.array([0.0]),  # knee == ankle
    )
    assert np.isnan(result2[0]), "Expected NaN for zero-length vector"


# ---------------------------------------------------------------------------
# Property 5: Proximity Score Bounds
# Validates: Requirements 3.1, 3.4
# ---------------------------------------------------------------------------

@given(distance=_distance)
@settings(max_examples=200)
def test_proximity_score_bounds(distance):
    """Property 5a: proximity score is always in [0, 100] for any distance >= 0."""
    result = compute_proximity_score(np.array([distance]))
    score = float(result[0])
    assert 0.0 <= score <= 100.0, f"score={score} out of [0, 100] for distance={distance}"


@given(distance=st.floats(min_value=5.0, max_value=1000.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=200)
def test_proximity_score_zero_beyond_max(distance):
    """Property 5b: proximity score is exactly 0 for distance >= 5.0 m."""
    result = compute_proximity_score(np.array([distance]))
    score = float(result[0])
    assert score == 0.0, f"Expected 0.0 for distance={distance}, got {score}"


# ---------------------------------------------------------------------------
# Property 6: PQI Composite Bounds
# Validates: Requirements 4.1, 4.5
# ---------------------------------------------------------------------------

@given(orientation=_score, stance=_score, proximity=_score)
@settings(max_examples=200)
def test_pqi_composite_bounds(orientation, stance, proximity):
    """Property 6: PQI is always in [0, 100] when all sub-scores are in [0, 100]."""
    result = compute_pqi(
        np.array([orientation]),
        np.array([stance]),
        np.array([proximity]),
    )
    pqi = float(result[0])
    assert 0.0 <= pqi <= 100.0, (
        f"PQI={pqi} out of [0, 100] for orientation={orientation}, stance={stance}, proximity={proximity}"
    )


# ===========================================================================
# Feature: football-body-intelligence-platform — Press Frame Run-Length
# ===========================================================================

from src.pressure_pipeline import identify_press_frames


def _check_run_lengths(result: pd.Series, min_run: int = 10) -> bool:
    """Every True in result must be in a run of >= min_run consecutive Trues."""
    if not result.any():
        return True
    arr = result.values
    run_id = np.cumsum(np.concatenate([[True], arr[:-1] != arr[1:]]))
    for rid in np.unique(run_id[arr]):
        if (run_id == rid).sum() < min_run:
            return False
    return True


# ---------------------------------------------------------------------------
# Property 7: Press Frame Run-Length Invariant
# Validates: Requirements 6.1, 6.5
# ---------------------------------------------------------------------------

@given(
    is_close_flags=st.lists(st.booleans(), min_size=1, max_size=200)
)
@settings(max_examples=200)
def test_press_frame_run_length_invariant(is_close_flags):
    """Property 7: every True in identify_press_frames output belongs to a run of >= 10 consecutive Trues.

    Validates: Requirements 6.1, 6.5
    """
    n = len(is_close_flags)
    frames = list(range(n))

    # Build presser_df: pelvis always at (0, 0)
    presser_df = pd.DataFrame({
        "frame_number": frames,
        "pelvis_x": np.zeros(n),
        "pelvis_y": np.zeros(n),
    })

    # Build carrier_df: close = 200 cm (2 m), far = 600 cm (6 m)
    distances_cm = np.where(is_close_flags, 200.0, 600.0)
    carrier_df = pd.DataFrame({
        "frame_number": frames,
        "pelvis_x": distances_cm,
        "pelvis_y": np.zeros(n),
    })

    result = identify_press_frames(presser_df, carrier_df)

    assert _check_run_lengths(result), (
        f"Found a True run shorter than 10 frames.\n"
        f"is_close_flags={is_close_flags}\n"
        f"result={result.tolist()}"
    )
