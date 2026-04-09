"""
pipelines/sagemaker_pipeline.py
Football Body Intelligence — SageMaker Processing orchestration via boto3.

Rewritten for SageMaker SDK v3 compatibility: uses boto3 directly instead of
the sagemaker.processing API (removed in v3). No SDK version dependency.

Pipeline structure (all match-level jobs run in parallel):

    AWI-FCB-HSV ─┐
    AWI-BVB-VFB ─┤
    AWI-SGE-FCB ─┤──► Aggregate-Results
    AWI-SGE-FCU ─┤       (waits for all 10 upstream jobs)
    AWI-FCU-FCB ─┤
    PQI-FCB-HSV ─┤
    PQI-BVB-VFB ─┤
    PQI-SGE-FCB ─┤
    PQI-SGE-FCU ─┤
    PQI-FCU-FCB ─┘

Usage:
    python pipelines/sagemaker_pipeline.py --action run
    python pipelines/sagemaker_pipeline.py --action status --run-id <timestamp>

Configuration via environment variables (never hardcoded):
    SM_IMAGE_URI        Full ECR image URI
    SM_ROLE_ARN         SageMaker execution role ARN
    S3_BUCKET           Data bucket name
    CHALLENGE_PREFIX    S3 prefix for match data (no trailing slash)
    AWS_DEFAULT_REGION  AWS region (default: eu-central-1)
    SM_INSTANCE_TYPE    Processing instance type (default: ml.m5.xlarge)
"""

import argparse
import os
import sys
import time
import datetime

import boto3


# ── Configuration (all from env vars, nothing hardcoded) ─────────────────────
REGION           = os.environ.get("AWS_DEFAULT_REGION", "eu-central-1")
IMAGE_URI        = os.environ.get("SM_IMAGE_URI", "")
ROLE_ARN         = os.environ.get("SM_ROLE_ARN", "")
S3_BUCKET        = os.environ.get("S3_BUCKET", "")
CHALLENGE_PREFIX = os.environ.get("CHALLENGE_PREFIX", "")
INSTANCE_TYPE    = os.environ.get("SM_INSTANCE_TYPE", "ml.m5.xlarge")

MATCH_IDS = ["FCB-HSV", "BVB-VFB", "SGE-FCB", "SGE-FCU", "FCU-FCB"]
S3_OUTPUT_BASE = f"s3://{S3_BUCKET}/pipeline-outputs"


def _assert_config() -> None:
    """Fail fast if required env vars are missing."""
    missing = [k for k, v in {
        "SM_IMAGE_URI":      IMAGE_URI,
        "SM_ROLE_ARN":       ROLE_ARN,
        "S3_BUCKET":         S3_BUCKET,
        "CHALLENGE_PREFIX":  CHALLENGE_PREFIX,
    }.items() if not v]
    if missing:
        print(f"ERROR: Missing required env vars: {missing}")
        print("Set them before running, e.g.:")
        for k in missing:
            print(f"  export {k}=<value>")
        sys.exit(1)


def _sm_client():
    return boto3.client("sagemaker", region_name=REGION)


def _submit_job(
    sm,
    job_name: str,
    script: str,
    env: dict,
    output_s3_uri: str,
) -> str:
    """Submit one SageMaker Processing job. Returns the job ARN."""
    response = sm.create_processing_job(
        ProcessingJobName=job_name,
        ProcessingResources={
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": INSTANCE_TYPE,
                "VolumeSizeInGB": 30,
            }
        },
        AppSpecification={
            "ImageUri": IMAGE_URI,
            # ContainerEntrypoint overrides CMD; script path is inside the image
            "ContainerEntrypoint": ["python3", script],
        },
        Environment=env,
        ProcessingOutputConfig={
            "Outputs": [{
                "OutputName": "output",
                "S3Output": {
                    "S3Uri": output_s3_uri,
                    "LocalPath": "/opt/ml/processing/output",
                    "S3UploadMode": "EndOfJob",
                },
            }]
        },
        RoleArn=ROLE_ARN,
    )
    return response["ProcessingJobArn"]


def _submit_aggregate_job(
    sm,
    job_name: str,
    run_id: str,
) -> str:
    """Submit the aggregation job with all match outputs as inputs."""
    inputs = []
    for match_id in MATCH_IDS:
        inputs.append({
            "InputName": f"awi-{match_id}",
            "S3Input": {
                "S3Uri": f"{S3_OUTPUT_BASE}/{run_id}/awi/{match_id}",
                "LocalPath": "/opt/ml/processing/input/awi",
                "S3DataType": "S3Prefix",
                "S3InputMode": "File",
            },
        })
        inputs.append({
            "InputName": f"pqi-{match_id}",
            "S3Input": {
                "S3Uri": f"{S3_OUTPUT_BASE}/{run_id}/pqi/{match_id}",
                "LocalPath": "/opt/ml/processing/input/pqi",
                "S3DataType": "S3Prefix",
                "S3InputMode": "File",
            },
        })

    response = sm.create_processing_job(
        ProcessingJobName=job_name,
        ProcessingResources={
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": "ml.m5.large",  # Aggregation is lightweight
                "VolumeSizeInGB": 10,
            }
        },
        AppSpecification={
            "ImageUri": IMAGE_URI,
            "ContainerEntrypoint": ["python3", "scripts/aggregate_results.py"],
        },
        Environment={
            "AWI_INPUT_DIR": "/opt/ml/processing/input/awi",
            "PQI_INPUT_DIR": "/opt/ml/processing/input/pqi",
        },
        ProcessingInputs=inputs,
        ProcessingOutputConfig={
            "Outputs": [{
                "OutputName": "final",
                "S3Output": {
                    "S3Uri": f"{S3_OUTPUT_BASE}/{run_id}/final",
                    "LocalPath": "/opt/ml/processing/output",
                    "S3UploadMode": "EndOfJob",
                },
            }]
        },
        RoleArn=ROLE_ARN,
    )
    return response["ProcessingJobArn"]


def _aggregate_locally(sm, run_id: str) -> None:
    """Download per-match CSVs from S3 and concatenate into awi_full / pqi_full.

    Used as fallback when the SageMaker aggregation job is blocked by an SCP.
    Writes directly to results/ in the repo root.
    """
    import glob
    import tempfile
    import pandas as pd

    s3 = boto3.client("s3", region_name=REGION)
    tmp = tempfile.mkdtemp(prefix="fbi-aggregate-")
    os.makedirs(f"{tmp}/awi", exist_ok=True)
    os.makedirs(f"{tmp}/pqi", exist_ok=True)

    print(f"  Downloading per-match CSVs from S3 to {tmp} ...")
    for match_id in MATCH_IDS:
        for metric in ("awi", "pqi"):
            prefix = f"pipeline-outputs/{run_id}/{metric}/{match_id}/"
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    filename = os.path.basename(key)
                    if filename.endswith(".csv"):
                        dest = f"{tmp}/{metric}/{filename}"
                        s3.download_file(S3_BUCKET, key, dest)
                        print(f"    Downloaded: {filename}")

    os.makedirs("results", exist_ok=True)
    for metric in ("awi", "pqi"):
        files = sorted(glob.glob(f"{tmp}/{metric}/*.csv"))
        if not files:
            print(f"  WARNING: No {metric.upper()} CSVs found — skipping.")
            continue
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        out = f"results/{metric}_full.csv"
        df.to_csv(out, index=False)
        print(f"  {metric.upper()}: {len(df)} rows saved to {out}")

    print(f"\nDone. Results saved to results/awi_full.csv and results/pqi_full.csv")


def _wait_for_jobs(sm, job_names: list[str], poll_interval: int = 30) -> set[str]:
    """Poll until all jobs reach a terminal state. Returns names of failed jobs."""
    terminal = {"Completed", "Failed", "Stopped"}
    print(f"\nMonitoring {len(job_names)} jobs (polling every {poll_interval}s)...")

    while True:
        statuses = {}
        for name in job_names:
            resp = sm.describe_processing_job(ProcessingJobName=name)
            statuses[name] = resp["ProcessingJobStatus"]

        n_done    = sum(1 for s in statuses.values() if s in terminal)
        n_running = len(job_names) - n_done
        n_failed  = sum(1 for s in statuses.values() if s in ("Failed", "Stopped"))

        print(
            f"  [{datetime.datetime.now().strftime('%H:%M:%S')}] "
            f"Running: {n_running} | Completed: {n_done - n_failed} | Failed: {n_failed}"
        )

        if n_done == len(job_names):
            failed = {n for n, s in statuses.items() if s in ("Failed", "Stopped")}
            if failed:
                print(f"\nFailed jobs: {sorted(failed)}")
                for name in failed:
                    resp = sm.describe_processing_job(ProcessingJobName=name)
                    print(f"  {name}: {resp.get('FailureReason', 'no reason given')}")
            return failed

        time.sleep(poll_interval)


def run_pipeline() -> None:
    """Submit all AWI + PQI jobs in parallel, then the aggregation job."""
    _assert_config()
    sm = _sm_client()

    # Unique run ID keeps job names and S3 paths isolated per run
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"Run ID: {run_id}")
    print(f"Instance type: {INSTANCE_TYPE}")
    print(f"Output prefix: {S3_OUTPUT_BASE}/{run_id}/")

    common_env = {
        "S3_BUCKET":          S3_BUCKET,
        "CHALLENGE_PREFIX":   CHALLENGE_PREFIX,
        "AWS_DEFAULT_REGION": REGION,
    }

    # ── Submit all 10 match jobs in parallel ─────────────────────────────────
    match_jobs: list[str] = []
    print("\nSubmitting AWI + PQI jobs...")
    for match_id in MATCH_IDS:
        for metric in ("awi", "pqi"):
            job_name = f"fbi-{metric}-{match_id.lower()}-{run_id}"
            script   = f"scripts/run_{metric}_job.py"
            output   = f"{S3_OUTPUT_BASE}/{run_id}/{metric}/{match_id}"
            env      = {**common_env, "MATCH_ID": match_id}

            _submit_job(sm, job_name, script, env, output)
            match_jobs.append(job_name)
            print(f"  Submitted: {job_name}")

    # ── Wait for all match jobs ───────────────────────────────────────────────
    failed = _wait_for_jobs(sm, match_jobs)
    if failed:
        print(f"\nERROR: {len(failed)} jobs failed. Check CloudWatch logs.")
        print("Re-run after fixing or skip failed matches in MATCH_CONFIGS.")
        sys.exit(1)

    print("\nAll match jobs completed. Aggregating results locally...")

    # ── Aggregate locally (SageMaker aggregation job may be blocked by SCP) ──
    _aggregate_locally(sm, run_id)


def show_status(run_id: str) -> None:
    """Print status of all jobs for a given run ID."""
    sm = _sm_client()
    paginator = sm.get_paginator("list_processing_jobs")
    for page in paginator.paginate(NameContains=f"fbi-"):
        for job in page["ProcessingJobSummaries"]:
            if run_id in job["ProcessingJobName"]:
                print(f"  {job['ProcessingJobName']:<55} {job['ProcessingJobStatus']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Football BI SageMaker Processing")
    parser.add_argument(
        "--action",
        choices=["run", "status"],
        default="run",
    )
    parser.add_argument(
        "--run-id",
        help="Run ID for --action status (format: YYYYMMDD-HHMMSS)",
        default=None,
    )
    args = parser.parse_args()

    if args.action == "run":
        run_pipeline()
    elif args.action == "status":
        if not args.run_id:
            print("ERROR: --run-id required for status action")
            sys.exit(1)
        show_status(args.run_id)


if __name__ == "__main__":
    main()
