"""
scripts/run_awi_job.py
SageMaker Processing entry point — AWI pipeline for a single match.

Each Processing Job handles exactly one match (identified via MATCH_ID env var).
The IAM execution role provides AWS credentials automatically — no explicit
key/secret needed. Output CSV is written to /opt/ml/processing/output/ and
uploaded to S3 by SageMaker after the job completes.

Environment variables (set in sagemaker_pipeline.py):
    MATCH_ID           e.g. "FCB-HSV"
    S3_BUCKET          e.g. "my-football-data-bucket"
    CHALLENGE_PREFIX   S3 prefix without trailing slash, e.g. "tracab/challenge"

Usage (local test, requires valid AWS profile):
    MATCH_ID=FCB-HSV S3_BUCKET=my-bucket CHALLENGE_PREFIX=tracab/challenge \
        python scripts/run_awi_job.py
"""

import os
import sys
import time

import boto3
import pyarrow.fs as pafs


# ── SageMaker Processing I/O directories ────────────────────────────────────
OUTPUT_DIR = os.environ.get("SM_OUTPUT_DIR", "/opt/ml/processing/output")


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


def main() -> None:
    # ── Read configuration from environment ─────────────────────────────────
    match_id         = os.environ["MATCH_ID"]
    bucket           = os.environ["S3_BUCKET"]
    challenge_prefix = os.environ["CHALLENGE_PREFIX"]
    region           = os.environ.get("AWS_DEFAULT_REGION", "eu-central-1")

    print(f"[AWI] Starting job: match={match_id} bucket={bucket} prefix={challenge_prefix}")
    t_start = time.time()

    # ── AWS clients (credentials from IAM role, no hardcoded keys) ──────────
    session   = boto3.Session(region_name=region)
    s3_client = session.client("s3")
    s3fs      = _build_s3fs(session, region)

    # ── Import existing pipeline code unchanged ──────────────────────────────
    # PYTHONPATH=/opt/ml/code set in Dockerfile, so src/ is importable.
    from src.batch_pipeline import MATCH_CONFIGS, run_match_awi  # noqa: E402

    # Resolve match config (raises StopIteration with a clear message if missing)
    try:
        match_config = next(m for m in MATCH_CONFIGS if m["match_id"] == match_id)
    except StopIteration:
        known = [m["match_id"] for m in MATCH_CONFIGS]
        print(f"[AWI] ERROR: MATCH_ID '{match_id}' not found. Known IDs: {known}")
        sys.exit(1)

    # ── Run pipeline ─────────────────────────────────────────────────────────
    # No checkpoint_path here: each job owns exactly one match, so a full
    # re-run on failure is cheap (~15-20 min) and simpler than shared checkpoints.
    df = run_match_awi(
        s3_client=s3_client,
        s3fs=s3fs,
        bucket=bucket,
        challenge_prefix=challenge_prefix,
        match_config=match_config,
        checkpoint_path=None,
    )

    if df.empty:
        print(f"[AWI] WARNING: No results produced for {match_id}.")
        sys.exit(1)

    # ── Write output ─────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"awi_{match_id}.csv")
    df.to_csv(out_path, index=False)

    elapsed = time.time() - t_start
    print(f"[AWI] Done: {len(df)} rows → {out_path} ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
