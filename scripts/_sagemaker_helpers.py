"""
scripts/_sagemaker_helpers.py

Shared scaffolding for the per-match SageMaker Processing entry points
(``run_awi_job.py`` and ``run_pqi_job.py``). Holds the pieces both metrics
need identically: env-var parsing, S3 client + PyArrow filesystem creation,
match-config lookup, and result CSV writing.

Each Processing Job handles exactly one match (identified via the ``MATCH_ID``
env var). The IAM execution role provides AWS credentials automatically - no
explicit key/secret needed. Output CSV is written to ``/opt/ml/processing/output/``
(or ``$SM_OUTPUT_DIR``) and uploaded to S3 by SageMaker after the job completes.

Required environment variables (set in ``pipelines/sagemaker_pipeline.py``):
    MATCH_ID           e.g. "FCB-HSV"
    S3_BUCKET          e.g. "my-football-data-bucket"
    CHALLENGE_PREFIX   S3 prefix without trailing slash, e.g. "tracab/challenge"

Optional:
    AWS_DEFAULT_REGION  default "eu-central-1"
    SM_OUTPUT_DIR       default "/opt/ml/processing/output"

Usage from a wrapper script:

    from scripts._sagemaker_helpers import run_metric_job

    if __name__ == "__main__":
        run_metric_job("awi")   # or "pqi"
"""

from __future__ import annotations

import os
import sys
import time
from typing import Callable, Literal

import boto3
import pandas as pd
import pyarrow.fs as pafs


OUTPUT_DIR = os.environ.get("SM_OUTPUT_DIR", "/opt/ml/processing/output")

# Type alias used to document the per-metric pipeline runner contract.
# Each runner must accept these kwargs and return a per-match result DataFrame.
_PipelineFn = Callable[..., pd.DataFrame]


def _build_s3fs(session: boto3.Session, region: str) -> pafs.S3FileSystem:
    """Build a PyArrow S3FileSystem from the current boto3 session credentials.

    SageMaker Processing provides temporary IAM role credentials via the EC2
    metadata service. Explicitly passing them to PyArrow ensures both libraries
    use the same identity and the credentials stay in sync on refresh.
    """
    creds = session.get_credentials().get_frozen_credentials()
    return pafs.S3FileSystem(
        access_key=creds.access_key,
        secret_key=creds.secret_key,
        session_token=creds.token,
        region=region,
    )


def _resolve_runner(metric: Literal["awi", "pqi"]) -> _PipelineFn:
    """Return the per-match runner for the requested metric.

    Imports are deferred so the wrapper scripts only pull in the heavy
    pipeline modules they actually need (and so import errors surface
    inside ``run_metric_job`` with a clean log line, not at module load).
    """
    if metric == "awi":
        from src.batch_pipeline import run_match_awi  # noqa: WPS433 (deferred import)
        return run_match_awi
    if metric == "pqi":
        from src.pressure_pipeline import run_match_pqi  # noqa: WPS433
        return run_match_pqi
    raise ValueError(f"Unknown metric '{metric}'. Expected 'awi' or 'pqi'.")


def run_metric_job(metric: Literal["awi", "pqi"]) -> None:
    """End-to-end driver for one per-match SageMaker Processing Job.

    Steps:
      1. Read MATCH_ID, S3_BUCKET, CHALLENGE_PREFIX, AWS_DEFAULT_REGION.
      2. Build a boto3 session, S3 client, and PyArrow S3FileSystem.
      3. Resolve the match config from ``src.batch_pipeline.MATCH_CONFIGS``.
      4. Invoke the per-metric runner.
      5. Write the result CSV to ``$SM_OUTPUT_DIR/{metric}_{MATCH_ID}.csv``.

    Exits with code 1 if MATCH_ID is unknown or the runner returns no rows.
    No checkpoint_path is passed: each job owns exactly one match, so a full
    re-run on failure (~15-20 min) is simpler than coordinating shared state.
    """
    tag = metric.upper()

    match_id         = os.environ["MATCH_ID"]
    bucket           = os.environ["S3_BUCKET"]
    challenge_prefix = os.environ["CHALLENGE_PREFIX"]
    region           = os.environ.get("AWS_DEFAULT_REGION", "eu-central-1")

    print(f"[{tag}] Starting job: match={match_id} bucket={bucket} prefix={challenge_prefix}")
    t_start = time.time()

    session   = boto3.Session(region_name=region)
    s3_client = session.client("s3")
    s3fs      = _build_s3fs(session, region)

    # MATCH_CONFIGS lives in batch_pipeline regardless of metric - it's the
    # single source of truth for the 5 challenge matches.
    from src.batch_pipeline import MATCH_CONFIGS  # noqa: WPS433
    runner = _resolve_runner(metric)

    try:
        match_config = next(m for m in MATCH_CONFIGS if m["match_id"] == match_id)
    except StopIteration:
        known = [m["match_id"] for m in MATCH_CONFIGS]
        print(f"[{tag}] ERROR: MATCH_ID '{match_id}' not found. Known IDs: {known}")
        sys.exit(1)

    df = runner(
        s3_client=s3_client,
        s3fs=s3fs,
        bucket=bucket,
        challenge_prefix=challenge_prefix,
        match_config=match_config,
        checkpoint_path=None,
    )

    if df.empty:
        print(f"[{tag}] WARNING: No results produced for {match_id}.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{metric}_{match_id}.csv")
    df.to_csv(out_path, index=False)

    elapsed = time.time() - t_start
    print(f"[{tag}] Done: {len(df)} rows -> {out_path} ({elapsed:.0f}s)")
