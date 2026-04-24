"""
scripts/run_awi_job.py

SageMaker Processing entry point for the AWI metric — one Processing Job per
match. The orchestrator (``pipelines/sagemaker_pipeline.py``) launches this
script with ``MATCH_ID`` set in the env. All shared logic lives in
``scripts/_sagemaker_helpers.py``.

Local test (requires a valid AWS profile):

    MATCH_ID=FCB-HSV S3_BUCKET=my-bucket CHALLENGE_PREFIX=tracab/challenge \\
        python scripts/run_awi_job.py
"""

import sys
from pathlib import Path

# Make sibling ``_sagemaker_helpers`` importable when run as a script.
# Inside the SageMaker container PYTHONPATH=/opt/ml/code already covers this.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._sagemaker_helpers import run_metric_job  # noqa: E402


if __name__ == "__main__":
    run_metric_job("awi")
