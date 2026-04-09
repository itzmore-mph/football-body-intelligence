import pytest
from unittest.mock import MagicMock, patch
import botocore.exceptions

import src.eda_helpers  # noqa: F401
from src.eda_helpers import create_session, list_bucket, SESSION_PROFILE, REGION


class TestCreateSession:
    def test_passes_profile_and_region_to_boto3(self):
        mock_session = MagicMock()
        mock_credentials = MagicMock()
        mock_credentials.access_key = "key"
        mock_credentials.secret_key = "secret"
        mock_credentials.token = "token"
        mock_session.get_credentials.return_value.get_frozen_credentials.return_value = mock_credentials
        mock_session.client.return_value = MagicMock()

        with patch("boto3.Session", return_value=mock_session) as mock_boto3_session:
            with patch("pyarrow.fs.S3FileSystem"):
                create_session(profile_name="my-profile", region="us-east-1")

        mock_boto3_session.assert_called_once_with(
            profile_name="my-profile", region_name="us-east-1"
        )

    def test_uses_default_profile_and_region(self):
        mock_session = MagicMock()
        mock_credentials = MagicMock()
        mock_credentials.access_key = "key"
        mock_credentials.secret_key = "secret"
        mock_credentials.token = "token"
        mock_session.get_credentials.return_value.get_frozen_credentials.return_value = mock_credentials
        mock_session.client.return_value = MagicMock()

        with patch("boto3.Session", return_value=mock_session) as mock_boto3_session:
            with patch("pyarrow.fs.S3FileSystem"):
                create_session()

        mock_boto3_session.assert_called_once_with(
            profile_name=SESSION_PROFILE, region_name=REGION
        )

    def test_returns_session_client_and_s3fs(self):
        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_credentials = MagicMock()
        mock_credentials.access_key = "key"
        mock_credentials.secret_key = "secret"
        mock_credentials.token = "token"
        mock_session.get_credentials.return_value.get_frozen_credentials.return_value = mock_credentials
        mock_session.client.return_value = mock_client
        mock_s3fs = MagicMock()

        with patch("boto3.Session", return_value=mock_session):
            with patch("pyarrow.fs.S3FileSystem", return_value=mock_s3fs) as mock_s3fs_cls:
                session, client, s3fs = create_session()

        assert session is mock_session
        assert client is mock_client
        assert s3fs is mock_s3fs
        # Verify credentials are passed through correctly to S3FileSystem
        mock_s3fs_cls.assert_called_once_with(
            access_key="key",
            secret_key="secret",
            session_token="token",
            region=REGION,
        )

    def test_profile_not_found_returns_none_tuple(self):
        with patch(
            "boto3.Session",
            side_effect=botocore.exceptions.ProfileNotFound(profile="missing-profile"),
        ):
            result = create_session(profile_name="missing-profile")

        assert result == (None, None, None)

    def test_profile_not_found_prints_profile_name(self, capsys):
        profile = "my-missing-profile"
        with patch(
            "boto3.Session",
            side_effect=botocore.exceptions.ProfileNotFound(profile=profile),
        ):
            create_session(profile_name=profile)

        captured = capsys.readouterr()
        assert profile in captured.out


class TestListBucket:
    def _make_client_error(self, code="AccessDenied", message="Access Denied"):
        return botocore.exceptions.ClientError(
            {"Error": {"Code": code, "Message": message}}, "ListObjectsV2"
        )

    def test_client_error_returns_empty_list(self):
        mock_client = MagicMock()
        mock_client.get_paginator.side_effect = self._make_client_error()

        result = list_bucket(mock_client, "my-bucket", "my-profile")

        assert result == []

    def test_client_error_output_contains_bucket_name(self, capsys):
        mock_client = MagicMock()
        mock_client.get_paginator.side_effect = self._make_client_error()

        list_bucket(mock_client, "my-bucket", "my-profile")

        assert "my-bucket" in capsys.readouterr().out

    def test_client_error_output_contains_profile(self, capsys):
        mock_client = MagicMock()
        mock_client.get_paginator.side_effect = self._make_client_error()

        list_bucket(mock_client, "my-bucket", "my-profile")

        assert "my-profile" in capsys.readouterr().out

    def test_client_error_uses_default_profile(self, capsys):
        mock_client = MagicMock()
        mock_client.get_paginator.side_effect = self._make_client_error()

        list_bucket(mock_client, "some-bucket")

        out = capsys.readouterr().out
        # SESSION_PROFILE comes from AWS_PROFILE env var; check whatever value is active
        assert str(SESSION_PROFILE) in out


from src.eda_helpers import load_json


class TestLoadJson:
    def _make_client_error(self, code="NoSuchKey", message="The specified key does not exist."):
        return botocore.exceptions.ClientError(
            {"Error": {"Code": code, "Message": message}}, "GetObject"
        )

    def test_no_such_key_returns_none(self):
        mock_client = MagicMock()
        mock_client.get_object.side_effect = self._make_client_error("NoSuchKey")

        result = load_json(mock_client, "my-bucket", "path/to/file.json")

        assert result is None

    def test_no_such_key_output_contains_s3_key(self, capsys):
        mock_client = MagicMock()
        mock_client.get_object.side_effect = self._make_client_error("NoSuchKey")

        load_json(mock_client, "my-bucket", "path/to/file.json")

        assert "path/to/file.json" in capsys.readouterr().out

    def test_malformed_json_returns_none(self):
        mock_client = MagicMock()
        mock_body = MagicMock()
        mock_body.read.return_value = b"not valid json {"
        mock_client.get_object.return_value = {"Body": mock_body}

        result = load_json(mock_client, "my-bucket", "path/to/file.json")

        assert result is None

    def test_malformed_json_output_contains_s3_key(self, capsys):
        mock_client = MagicMock()
        mock_body = MagicMock()
        mock_body.read.return_value = b"not valid json {"
        mock_client.get_object.return_value = {"Body": mock_body}

        load_json(mock_client, "my-bucket", "path/to/file.json")

        out = capsys.readouterr().out
        assert "path/to/file.json" in out

    def test_malformed_json_output_contains_parse_error(self, capsys):
        mock_client = MagicMock()
        mock_body = MagicMock()
        mock_body.read.return_value = b"not valid json {"
        mock_client.get_object.return_value = {"Body": mock_body}

        load_json(mock_client, "my-bucket", "path/to/file.json")

        out = capsys.readouterr().out
        # The JSONDecodeError message should appear in the output
        assert "parse" in out.lower() or "json" in out.lower() or "error" in out.lower()


from src.eda_helpers import load_xml
import xml.etree.ElementTree as ET


class TestLoadXml:
    def _make_client_error(self, code="NoSuchKey", message="The specified key does not exist."):
        return botocore.exceptions.ClientError(
            {"Error": {"Code": code, "Message": message}}, "GetObject"
        )

    def test_client_error_returns_none(self):
        mock_client = MagicMock()
        mock_client.get_object.side_effect = self._make_client_error("NoSuchKey")

        result = load_xml(mock_client, "my-bucket", "path/to/file.xml")

        assert result is None

    def test_client_error_output_contains_s3_key(self, capsys):
        mock_client = MagicMock()
        mock_client.get_object.side_effect = self._make_client_error("NoSuchKey")

        load_xml(mock_client, "my-bucket", "path/to/file.xml")

        assert "path/to/file.xml" in capsys.readouterr().out

    def test_malformed_xml_returns_none(self):
        mock_client = MagicMock()
        mock_body = MagicMock()
        mock_body.read.return_value = b"<unclosed>"
        mock_client.get_object.return_value = {"Body": mock_body}

        result = load_xml(mock_client, "my-bucket", "path/to/file.xml")

        assert result is None

    def test_malformed_xml_output_contains_s3_key(self, capsys):
        mock_client = MagicMock()
        mock_body = MagicMock()
        mock_body.read.return_value = b"<unclosed>"
        mock_client.get_object.return_value = {"Body": mock_body}

        load_xml(mock_client, "my-bucket", "path/to/file.xml")

        assert "path/to/file.xml" in capsys.readouterr().out

    def test_malformed_xml_output_contains_parse_error(self, capsys):
        mock_client = MagicMock()
        mock_body = MagicMock()
        mock_body.read.return_value = b"<unclosed>"
        mock_client.get_object.return_value = {"Body": mock_body}

        load_xml(mock_client, "my-bucket", "path/to/file.xml")

        out = capsys.readouterr().out
        assert "parse" in out.lower() or "xml" in out.lower() or "error" in out.lower()


from src.eda_helpers import sample_parquet
import pyarrow as pa


class TestSampleParquet:
    def test_iter_batches_called_with_batch_size(self):
        mock_fs = MagicMock()
        mock_batch = pa.record_batch({"col": [1, 2, 3]})
        mock_pf = MagicMock()
        mock_pf.iter_batches.return_value = iter([mock_batch])

        with patch("pyarrow.parquet.ParquetFile", return_value=mock_pf):
            result = sample_parquet(mock_fs, "bucket/file.parquet", nrows=10_000)

        mock_pf.iter_batches.assert_called_once_with(batch_size=10_000)
        assert result is not None
        assert len(result) == 3

    def test_iter_batches_uses_custom_nrows(self):
        mock_fs = MagicMock()
        mock_batch = pa.record_batch({"col": [1]})
        mock_pf = MagicMock()
        mock_pf.iter_batches.return_value = iter([mock_batch])

        with patch("pyarrow.parquet.ParquetFile", return_value=mock_pf):
            sample_parquet(mock_fs, "bucket/file.parquet", nrows=500)

        mock_pf.iter_batches.assert_called_once_with(batch_size=500)

    def test_error_path_returns_none(self):
        mock_fs = MagicMock()

        with patch("pyarrow.parquet.ParquetFile", side_effect=RuntimeError("read failed")):
            result = sample_parquet(mock_fs, "bucket/file.parquet")

        assert result is None

    def test_error_path_output_contains_s3_path(self, capsys):
        mock_fs = MagicMock()
        path = "bucket/some/file.parquet"

        with patch("pyarrow.parquet.ParquetFile", side_effect=RuntimeError("read failed")):
            sample_parquet(mock_fs, path)

        assert path in capsys.readouterr().out

    def test_error_path_output_contains_exception_repr(self, capsys):
        mock_fs = MagicMock()
        exc = RuntimeError("read failed")

        with patch("pyarrow.parquet.ParquetFile", side_effect=exc):
            sample_parquet(mock_fs, "bucket/file.parquet")

        assert repr(exc) in capsys.readouterr().out
