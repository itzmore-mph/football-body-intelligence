"""
pipeline_io.py

Shared S3 / Parquet IO helpers used by both ``batch_pipeline`` (AWI) and
``pressure_pipeline`` (PQI). Two responsibilities:

1. Reading TF15 phase boundaries from Parquet KV metadata.
2. Reading per-phase parquet tables with retry-on-transient-error semantics
   that surface AWS SSO token expiry immediately (no wasted backoff time).

Why a separate module: both pipelines need identical retry behavior against
the same S3 objects, but they read different column subsets. Centralising the
retry loop avoids drift, while letting each caller specify its own columns.

Token-expiry markers are matched against the exception string. If any of these
substrings is present, the helper re-raises immediately, since the only fix is
``aws sso login``. Other errors (transient network, throttling) are retried
with exponential backoff: ``backoff_base * 2 ** attempt`` seconds.
"""

from __future__ import annotations

import time
from typing import Sequence

import pandas as pd
import pyarrow.fs as pafs
import pyarrow.parquet as pq

from src.event_parser import extract_phases_from_metadata


# Substrings that indicate the AWS SSO token has expired. These are not
# transient: re-running with a fresh token is the only fix, so we re-raise
# rather than waste backoff time.
_TOKEN_ERROR_MARKERS = ("ExpiredToken", "InvalidClientTokenId", "HTTP 400", "400")


def load_phases_from_parquet(
    s3fs: pafs.S3FileSystem,
    parquet_path: str,
) -> list[dict]:
    """Read phase frame boundaries from the TF15 Parquet metadata header.

    The TF15 1.1 spec stores phase boundaries as key-value metadata on the
    Parquet file itself (not in a separate JSON). Keys: phase_1_start,
    phase_1_end, phase_2_start, phase_2_end (bytes when read via pyarrow).

    Args:
        s3fs:         pyarrow S3FileSystem.
        parquet_path: Full S3 path without ``s3://`` prefix (bucket/key).

    Returns:
        List of phase dicts from :func:`event_parser.extract_phases_from_metadata`.

    Raises:
        RuntimeError: If TF15 metadata is missing or unrecognized.
    """
    pf = pq.ParquetFile(parquet_path, filesystem=s3fs)
    kv = pf.metadata.metadata  # dict[bytes, bytes] or None
    if not kv:
        raise RuntimeError(
            f"Parquet file has no key-value metadata: {parquet_path}. "
            "Cannot determine phase boundaries automatically."
        )
    return extract_phases_from_metadata(dict(kv))


def read_phase_table_with_retry(
    s3fs: pafs.S3FileSystem,
    parquet_path: str,
    phase_start: int,
    phase_end: int,
    columns: Sequence[str],
    max_retries: int = 3,
    backoff_base: float = 15.0,
    log_prefix: str = "    ",
):
    """Read a parquet phase window with column pruning and retry on transient errors.

    Filter is applied at row-group granularity (a few frames outside
    ``[phase_start, phase_end]`` may be included; the metric callers clip them).

    Token-expiry errors are re-raised immediately (no retry, no wait), since
    they require ``aws sso login`` to fix. All other exceptions are retried up
    to ``max_retries`` times with exponential backoff
    (``backoff_base * 2 ** attempt`` seconds: 15, 30, 60s by default).

    Args:
        s3fs:         pyarrow S3FileSystem.
        parquet_path: Full S3 path (bucket/challenge_prefix/match/file.parquet).
        phase_start:  Inclusive start frame number.
        phase_end:    Inclusive end frame number.
        columns:      Parquet column names to read. Subset to reduce memory
                      pressure; AWI typically wants
                      ``["frame_number", "skeletons"]``, PQI also needs
                      ``"ball"`` and ``"ball_exists"``.
        max_retries:  Total attempts before giving up (default 3).
        backoff_base: Base wait in seconds before doubling (default 15).
        log_prefix:   String prepended to each retry log line.

    Returns:
        ``pyarrow.Table`` for the given phase window.

    Raises:
        Exception: The last underlying exception if all retries are exhausted,
                   or the original exception immediately on token expiry.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return pq.read_table(
                parquet_path,
                filesystem=s3fs,
                columns=list(columns),
                filters=[
                    ("frame_number", ">=", phase_start),
                    ("frame_number", "<=", phase_end),
                ],
            )
        except Exception as e:
            err_str = str(e)
            if any(marker in err_str for marker in _TOKEN_ERROR_MARKERS):
                print(f"{log_prefix}[token expired] {e}. Refresh token and re-run.")
                raise
            last_exc = e
            wait = backoff_base * (2 ** attempt)
            print(
                f"{log_prefix}[retry {attempt + 1}/{max_retries}] {e}. "
                f"Waiting {wait:.0f}s..."
            )
            time.sleep(wait)
    raise last_exc  # type: ignore[misc]


def load_phase_df(
    s3fs: pafs.S3FileSystem,
    parquet_path: str,
    phase_start: int,
    phase_end: int,
) -> pd.DataFrame:
    """Stream-read skeleton frames for a single phase from S3.

    Convenience wrapper around :func:`read_phase_table_with_retry` that returns
    a pandas DataFrame and reads only the columns AWI needs
    (``frame_number`` + ``skeletons``). For PQI use
    :func:`read_phase_table_with_retry` directly with the wider column list.
    """
    table = read_phase_table_with_retry(
        s3fs,
        parquet_path,
        phase_start,
        phase_end,
        columns=["frame_number", "skeletons"],
    )
    return table.to_pandas()
