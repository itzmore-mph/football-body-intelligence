"""
s3_data_loader.py

Loads CSV result files from S3 for cloud-hosted dashboard deployments
(e.g. Streamlit Community Cloud). Falls back to local files for development.

Credentials are provided via:
  - Streamlit secrets (st.secrets["aws"]) when deployed on Streamlit Cloud
  - Environment variables / AWS profile when running locally
"""

import io
import logging
import os
from typing import Optional

import boto3
import pandas as pd
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)

# S3 prefix where result CSVs are stored
_DEFAULT_S3_RESULTS_PREFIX = "results"


def _get_streamlit_secrets() -> Optional[dict]:
    """Attempt to read AWS config from Streamlit secrets.

    Returns the aws secrets dict or None if unavailable.
    """
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "aws" in st.secrets:
            return dict(st.secrets["aws"])
    except Exception:
        pass
    return None


def _get_s3_config() -> Optional[dict]:
    """Extract S3 configuration from Streamlit secrets or environment.

    Returns dict with keys: bucket, prefix, aws_access_key_id,
    aws_secret_access_key, region_name. Returns None if not configured.
    """
    # Try Streamlit secrets first (used on Streamlit Community Cloud)
    aws_conf = _get_streamlit_secrets()
    if aws_conf is not None:
        return {
            "bucket": aws_conf["bucket"],
            "prefix": aws_conf.get("results_prefix", _DEFAULT_S3_RESULTS_PREFIX),
            "aws_access_key_id": aws_conf["aws_access_key_id"],
            "aws_secret_access_key": aws_conf["aws_secret_access_key"],
            "region_name": aws_conf.get("region_name", "eu-central-1"),
        }

    # Fall back to environment variables
    bucket = os.environ.get("HACKATHON_BUCKET")
    if bucket:
        return {
            "bucket": bucket,
            "prefix": os.environ.get("S3_RESULTS_PREFIX", _DEFAULT_S3_RESULTS_PREFIX),
            "aws_access_key_id": os.environ.get("AWS_ACCESS_KEY_ID"),
            "aws_secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY"),
            "region_name": os.environ.get("AWS_DEFAULT_REGION", "eu-central-1"),
        }

    return None


def _create_s3_client(config: dict):
    """Create a boto3 S3 client from config dict."""
    kwargs = {"region_name": config["region_name"]}
    if config.get("aws_access_key_id") and config.get("aws_secret_access_key"):
        kwargs["aws_access_key_id"] = config["aws_access_key_id"]
        kwargs["aws_secret_access_key"] = config["aws_secret_access_key"]
    return boto3.client("s3", **kwargs)


def read_csv_from_s3(filename: str) -> Optional[pd.DataFrame]:
    """Read a CSV file from S3.

    Args:
        filename: Name of the CSV file (e.g. "awi_full.csv").

    Returns:
        DataFrame if successful, None if S3 is not configured or file not found.
    """
    config = _get_s3_config()
    if config is None:
        return None

    key = f"{config['prefix']}/{filename}"
    try:
        client = _create_s3_client(config)
        response = client.get_object(Bucket=config["bucket"], Key=key)
        body = response["Body"].read()
        return pd.read_csv(io.BytesIO(body))
    except (ClientError, NoCredentialsError, KeyError) as e:
        logger.warning("Could not read s3://%s/%s: %s", config["bucket"], key, e)
        return None
    except Exception as e:
        logger.warning("Unexpected error reading from S3: %s", e)
        return None


def load_csv(filename: str, local_path: str) -> Optional[pd.DataFrame]:
    """Load a CSV file, trying S3 first then falling back to local disk.

    Args:
        filename: CSV filename (e.g. "awi_full.csv").
        local_path: Local file path to fall back to (e.g. "results/awi_full.csv").

    Returns:
        DataFrame if found from either source, None if neither available.
    """
    # Try S3 first
    df = read_csv_from_s3(filename)
    if df is not None:
        logger.info("Loaded %s from S3", filename)
        return df

    # Fall back to local file
    if os.path.exists(local_path):
        logger.info("Loaded %s from local disk", local_path)
        return pd.read_csv(local_path)

    return None
