"""
tests/test_pipeline_io.py

Unit tests for src/pipeline_io.py

All S3/Parquet IO is mocked -- no network access required.

Covers:
  load_phases_from_parquet:
    - Delegates to extract_phases_from_metadata with decoded KV metadata
    - Raises RuntimeError when Parquet file has no KV metadata

  read_phase_table_with_retry:
    - Succeeds on first attempt
    - Retries on transient errors and succeeds
    - Re-raises immediately on token-expiry errors (no retry)
    - Raises after exhausting all retries

  load_phase_df:
    - Returns a pandas DataFrame with correct columns
    - Passes correct column list to read_phase_table_with_retry
"""

from unittest.mock import MagicMock, patch

import pyarrow as pa
import pytest

from src.pipeline_io import (
    _TOKEN_ERROR_MARKERS,
    load_phase_df,
    load_phases_from_parquet,
    read_phase_table_with_retry,
)


class TestLoadPhasesFromParquet:

    @patch("src.pipeline_io.pq.ParquetFile")
    @patch("src.pipeline_io.extract_phases_from_metadata")
    def test_delegates_to_extract_phases(self, mock_extract, mock_pf_cls):
        fake_kv = {b"phase_1_start": b"100", b"phase_1_end": b"200"}
        mock_pf = MagicMock()
        mock_pf.metadata.metadata = fake_kv
        mock_pf_cls.return_value = mock_pf
        mock_extract.return_value = [{"section": "1", "start_frame": 100, "end_frame": 200}]

        result = load_phases_from_parquet(MagicMock(), "bucket/key.parquet")

        mock_extract.assert_called_once_with({b"phase_1_start": b"100", b"phase_1_end": b"200"})
        assert len(result) == 1
        assert result[0]["start_frame"] == 100

    @patch("src.pipeline_io.pq.ParquetFile")
    def test_raises_on_missing_metadata(self, mock_pf_cls):
        mock_pf = MagicMock()
        mock_pf.metadata.metadata = None
        mock_pf_cls.return_value = mock_pf

        with pytest.raises(RuntimeError, match="no key-value metadata"):
            load_phases_from_parquet(MagicMock(), "bucket/key.parquet")


class TestReadPhaseTableWithRetry:

    @patch("src.pipeline_io.pq.read_table")
    def test_success_on_first_attempt(self, mock_read):
        fake_table = pa.table({"frame_number": [100, 101], "skeletons": [None, None]})
        mock_read.return_value = fake_table

        result = read_phase_table_with_retry(
            MagicMock(), "path.parquet", 100, 200, ["frame_number", "skeletons"]
        )
        assert result.num_rows == 2
        mock_read.assert_called_once()

    @patch("src.pipeline_io.time.sleep")
    @patch("src.pipeline_io.pq.read_table")
    def test_retries_on_transient_error(self, mock_read, mock_sleep):
        fake_table = pa.table({"frame_number": [100]})
        mock_read.side_effect = [ConnectionError("network blip"), fake_table]

        result = read_phase_table_with_retry(
            MagicMock(), "path.parquet", 100, 200, ["frame_number"],
            max_retries=3, backoff_base=1.0,
        )
        assert result.num_rows == 1
        assert mock_read.call_count == 2
        mock_sleep.assert_called_once_with(1.0)  # backoff_base * 2^0

    @patch("src.pipeline_io.time.sleep")
    @patch("src.pipeline_io.pq.read_table")
    def test_token_expiry_raises_immediately(self, mock_read, mock_sleep):
        for marker in _TOKEN_ERROR_MARKERS[:2]:  # Test first two markers
            mock_read.side_effect = Exception(f"Error: {marker} in request")
            mock_sleep.reset_mock()

            with pytest.raises(Exception, match=marker):
                read_phase_table_with_retry(
                    MagicMock(), "path.parquet", 100, 200, ["frame_number"],
                    max_retries=3, backoff_base=1.0,
                )
            # Should NOT have slept (no retry on token expiry)
            mock_sleep.assert_not_called()

    @patch("src.pipeline_io.time.sleep")
    @patch("src.pipeline_io.pq.read_table")
    def test_raises_after_exhausting_retries(self, mock_read, mock_sleep):
        mock_read.side_effect = ConnectionError("persistent failure")

        with pytest.raises(ConnectionError, match="persistent failure"):
            read_phase_table_with_retry(
                MagicMock(), "path.parquet", 100, 200, ["frame_number"],
                max_retries=3, backoff_base=0.01,
            )
        assert mock_read.call_count == 3
        assert mock_sleep.call_count == 3

    @patch("src.pipeline_io.pq.read_table")
    def test_passes_correct_filters_and_columns(self, mock_read):
        fake_table = pa.table({"frame_number": [100], "skeletons": [None]})
        mock_read.return_value = fake_table
        mock_fs = MagicMock()

        read_phase_table_with_retry(
            mock_fs, "bucket/file.parquet", 500, 1000,
            ["frame_number", "skeletons"],
        )

        mock_read.assert_called_once_with(
            "bucket/file.parquet",
            filesystem=mock_fs,
            columns=["frame_number", "skeletons"],
            filters=[
                ("frame_number", ">=", 500),
                ("frame_number", "<=", 1000),
            ],
        )


class TestLoadPhaseDf:

    @patch("src.pipeline_io.read_phase_table_with_retry")
    def test_returns_dataframe_with_correct_columns(self, mock_retry):
        fake_table = pa.table({
            "frame_number": [100, 101, 102],
            "skeletons": [None, None, None],
        })
        mock_retry.return_value = fake_table

        result = load_phase_df(MagicMock(), "path.parquet", 100, 102)

        assert list(result.columns) == ["frame_number", "skeletons"]
        assert len(result) == 3

    @patch("src.pipeline_io.read_phase_table_with_retry")
    def test_passes_awi_columns(self, mock_retry):
        fake_table = pa.table({"frame_number": [1], "skeletons": [None]})
        mock_retry.return_value = fake_table
        mock_fs = MagicMock()

        load_phase_df(mock_fs, "path.parquet", 1, 100)

        mock_retry.assert_called_once_with(
            mock_fs, "path.parquet", 1, 100,
            columns=["frame_number", "skeletons"],
        )
