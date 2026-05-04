# EDA helper functions for football 3D data analysis

import json
import os
import xml.etree.ElementTree as ET
import defusedxml.ElementTree as _safe_ET

import boto3
import botocore.exceptions
import pandas as pd
import pyarrow.fs as pafs
import pyarrow.parquet as pq

# Read from environment - set AWS_PROFILE before running (e.g. in .env or shell export).
# boto3 also respects AWS_PROFILE natively, so this just makes the default explicit.
SESSION_PROFILE = os.environ.get("AWS_PROFILE")
REGION = os.environ.get("AWS_DEFAULT_REGION", "eu-central-1")


def create_session(
    profile_name: str = SESSION_PROFILE,
    region: str = REGION,
) -> tuple:
    """Create a boto3 Session, S3 client, and pyarrow S3FileSystem.

    Returns:
        (boto3.Session, boto3.client, pyarrow.fs.S3FileSystem)
    """
    try:
        session = boto3.Session(profile_name=profile_name, region_name=region)
    except botocore.exceptions.ProfileNotFound:
        print(
            f"[ERROR] AWS profile '{profile_name}' not found. "
            "Please configure it in ~/.aws/credentials or ~/.aws/config."
        )
        return None, None, None

    s3_client = session.client("s3")

    # SSO credentials are refreshable - resolve lazily via refresh()
    # .resolve() is not available on DeferredRefreshableCredentials
    raw_creds = session.get_credentials()
    if raw_creds is None:
        print(
            f"[ERROR] No AWS credentials found for profile '{profile_name}'. "
            "Run `aws sso login --profile <profile>` or set AWS_PROFILE in your environment."
        )
        return None, None, None
    try:
        credentials = raw_creds.get_frozen_credentials()
    except Exception as e:
        print(f"[ERROR] Unexpected error creating AWS session: {e}")
        return None, None, None

    s3fs = pafs.S3FileSystem(
        access_key=credentials.access_key,
        secret_key=credentials.secret_key,
        session_token=credentials.token,
        region=region,
    )

    return session, s3_client, s3fs


def list_bucket(
    client,
    bucket: str,
    profile: str = SESSION_PROFILE,
) -> list[dict]:
    """List all objects in an S3 bucket, handling pagination automatically.

    Args:
        client: boto3 S3 client
        bucket: S3 bucket name
        profile: AWS profile name (used in error messages)

    Returns:
        Flat list of S3 object dicts (each has at least 'Key' and 'Size').
    """
    try:
        paginator = client.get_paginator("list_objects_v2")
        objects = []
        for page in paginator.paginate(Bucket=bucket):
            objects.extend(page.get("Contents", []))
        return objects
    except botocore.exceptions.ClientError as e:
        error_code = e.response["Error"]["Code"]
        error_msg = e.response["Error"]["Message"]
        print(
            f"[ERROR] Failed to list bucket '{bucket}' using profile '{profile}': "
            f"{error_code} - {error_msg}"
        )
        return []


def load_json(client, bucket: str, key: str) -> dict | None:
    """Load and parse a JSON file from S3.

    Args:
        client: boto3 S3 client
        bucket: S3 bucket name
        key: S3 object key

    Returns:
        Parsed dict, or None on error.
    """
    try:
        response = client.get_object(Bucket=bucket, Key=key)
        return json.loads(response["Body"].read())
    except botocore.exceptions.ClientError as e:
        error_code = e.response["Error"]["Code"]
        print(f"[ERROR] Failed to load s3://{bucket}/{key}: {error_code}")
        return None
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse JSON from s3://{bucket}/{key}: {e}")
        return None


def load_xml(client, bucket: str, key: str) -> ET.Element | None:
    """Load and parse an XML file from S3.

    Args:
        client: boto3 S3 client
        bucket: S3 bucket name
        key: S3 object key

    Returns:
        Parsed ET.Element root, or None on error.
    """
    try:
        response = client.get_object(Bucket=bucket, Key=key)
        return _safe_ET.fromstring(response["Body"].read())
    except botocore.exceptions.ClientError as e:
        error_code = e.response["Error"]["Code"]
        print(f"[ERROR] Failed to load s3://{bucket}/{key}: {error_code}")
        return None
    except ET.ParseError as e:
        print(f"[ERROR] Failed to parse XML from s3://{bucket}/{key}: {e}")
        return None


def sample_parquet(fs: pafs.S3FileSystem, path: str, nrows: int = 10_000) -> pd.DataFrame | None:
    """Sample the first `nrows` rows from a Parquet file on S3.

    Args:
        fs: pyarrow S3FileSystem
        path: S3 path to the Parquet file (without s3:// prefix)
        nrows: number of rows to read (default 10,000)

    Returns:
        pandas DataFrame with up to `nrows` rows, or None on error.
    """
    try:
        pf = pq.ParquetFile(path, filesystem=fs)
        batch = next(pf.iter_batches(batch_size=nrows))
        return batch.to_pandas()
    except Exception as e:
        print(f"[ERROR] Failed to sample parquet at {path}: {repr(e)}")
        return None


def report_dataframe(name: str, df: pd.DataFrame, schema=None):
    """Print a labelled summary of a pandas DataFrame.

    Args:
        name: Display name / header for the report section.
        df: The DataFrame to report on.
        schema: Optional pyarrow schema to print before shape/dtypes.
    """
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    if schema:
        print("Parquet schema:\n", schema)
    print("Shape:", df.shape)
    print("Dtypes:\n", df.dtypes)
    print("First 5 rows:\n", df.head())


def report_dict(name: str, d: dict):
    """Print a labelled summary of a Python dictionary.

    Args:
        name: Display name / header for the report section.
        d: The dictionary to report on.
    """
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    for k, v in d.items():
        print(f"  {k}: {type(v).__name__} = {v}")


def report_xml(name: str, root: ET.Element):
    """Print a labelled summary of an XML element tree.

    Args:
        name: Display name / header for the report section.
        root: The root ET.Element to inspect.
    """
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    print("Root tag:", root.tag)
    print("Root attributes:", root.attrib)
    children = list(root)
    print("Child tags:", [c.tag for c in children])
    print("Child count:", len(children))
