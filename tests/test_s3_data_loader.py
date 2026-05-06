"""Tests for src/s3_data_loader.py"""

import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.s3_data_loader import _get_s3_config, load_csv, read_csv_from_s3


class TestGetS3Config:
    """Tests for _get_s3_config helper."""

    def test_returns_none_when_no_config(self, monkeypatch):
        monkeypatch.delenv("HACKATHON_BUCKET", raising=False)
        with patch("src.s3_data_loader._get_streamlit_secrets", return_value=None):
            result = _get_s3_config()
        assert result is None

    def test_reads_from_env(self, monkeypatch):
        monkeypatch.setenv("HACKATHON_BUCKET", "my-bucket")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIATEST")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret123")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
        with patch("src.s3_data_loader._get_streamlit_secrets", return_value=None):
            result = _get_s3_config()
        assert result is not None
        assert result["bucket"] == "my-bucket"
        assert result["aws_access_key_id"] == "AKIATEST"
        assert result["region_name"] == "us-east-1"
        assert result["prefix"] == "results"

    def test_reads_from_streamlit_secrets(self, monkeypatch):
        monkeypatch.delenv("HACKATHON_BUCKET", raising=False)
        secrets = {
            "bucket": "st-bucket",
            "aws_access_key_id": "AKIAST",
            "aws_secret_access_key": "stsecret",
            "region_name": "eu-west-1",
        }
        with patch("src.s3_data_loader._get_streamlit_secrets", return_value=secrets):
            result = _get_s3_config()
        assert result is not None
        assert result["bucket"] == "st-bucket"
        assert result["aws_access_key_id"] == "AKIAST"
        assert result["region_name"] == "eu-west-1"

    def test_streamlit_secrets_take_precedence(self, monkeypatch):
        monkeypatch.setenv("HACKATHON_BUCKET", "env-bucket")
        secrets = {
            "bucket": "st-bucket",
            "aws_access_key_id": "AKIAST",
            "aws_secret_access_key": "stsecret",
        }
        with patch("src.s3_data_loader._get_streamlit_secrets", return_value=secrets):
            result = _get_s3_config()
        assert result["bucket"] == "st-bucket"


class TestReadCsvFromS3:
    """Tests for read_csv_from_s3."""

    def test_returns_none_when_no_config(self, monkeypatch):
        monkeypatch.delenv("HACKATHON_BUCKET", raising=False)
        with patch("src.s3_data_loader._get_streamlit_secrets", return_value=None):
            result = read_csv_from_s3("test.csv")
        assert result is None

    def test_returns_dataframe_on_success(self, monkeypatch):
        monkeypatch.setenv("HACKATHON_BUCKET", "test-bucket")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIATEST")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")

        csv_content = b"col1,col2\n1,2\n3,4\n"
        mock_body = MagicMock()
        mock_body.read.return_value = csv_content

        with patch("src.s3_data_loader._get_streamlit_secrets", return_value=None), \
             patch("src.s3_data_loader.boto3") as mock_boto3:
            mock_client = MagicMock()
            mock_client.get_object.return_value = {"Body": mock_body}
            mock_boto3.client.return_value = mock_client

            result = read_csv_from_s3("test.csv")

        assert result is not None
        assert list(result.columns) == ["col1", "col2"]
        assert len(result) == 2
        mock_client.get_object.assert_called_once_with(
            Bucket="test-bucket", Key="results/test.csv"
        )

    def test_returns_none_on_client_error(self, monkeypatch):
        monkeypatch.setenv("HACKATHON_BUCKET", "test-bucket")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIATEST")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")

        from botocore.exceptions import ClientError

        with patch("src.s3_data_loader._get_streamlit_secrets", return_value=None), \
             patch("src.s3_data_loader.boto3") as mock_boto3:
            mock_client = MagicMock()
            mock_client.get_object.side_effect = ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "Not found"}},
                "GetObject",
            )
            mock_boto3.client.return_value = mock_client

            result = read_csv_from_s3("missing.csv")

        assert result is None


class TestLoadCsv:
    """Tests for load_csv with S3 + local fallback."""

    def test_falls_back_to_local(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HACKATHON_BUCKET", raising=False)
        local_file = tmp_path / "test.csv"
        local_file.write_text("a,b\n1,2\n")

        with patch("src.s3_data_loader._get_streamlit_secrets", return_value=None):
            result = load_csv("test.csv", str(local_file))

        assert result is not None
        assert list(result.columns) == ["a", "b"]

    def test_returns_none_when_nothing_available(self, monkeypatch):
        monkeypatch.delenv("HACKATHON_BUCKET", raising=False)
        with patch("src.s3_data_loader._get_streamlit_secrets", return_value=None):
            result = load_csv("nonexistent.csv", "/no/such/path.csv")
        assert result is None

    def test_prefers_s3_over_local(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HACKATHON_BUCKET", "test-bucket")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIATEST")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")

        # Local file has different data
        local_file = tmp_path / "test.csv"
        local_file.write_text("a,b\nlocal,data\n")

        # S3 returns different data
        csv_content = b"a,b\ns3,data\n"
        mock_body = MagicMock()
        mock_body.read.return_value = csv_content

        with patch("src.s3_data_loader._get_streamlit_secrets", return_value=None), \
             patch("src.s3_data_loader.boto3") as mock_boto3:
            mock_client = MagicMock()
            mock_client.get_object.return_value = {"Body": mock_body}
            mock_boto3.client.return_value = mock_client

            result = load_csv("test.csv", str(local_file))

        assert result is not None
        assert result.iloc[0]["a"] == "s3"
