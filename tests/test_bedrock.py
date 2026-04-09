"""
tests/test_bedrock.py

Unit tests for src/bedrock_client.py.
All tests use mocks — no real AWS calls.

Validates: Requirements 10.1, 10.2
"""

import io
import json
import time
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from botocore.exceptions import ClientError

from src.bedrock_client import (
    BedrockClientError,
    batch_generate_narratives,
    build_player_prompt,
    generate_player_narrative,
)

# ---------------------------------------------------------------------------
# Shared sample data
# ---------------------------------------------------------------------------

SAMPLE_PLAYER_ROW = {
    "name": "Thomas Müller",
    "position": "AM",
    "match_id": "match_001",
    "phase_label": "build_up",
    "awi_per_minute": 12.5,
}

SAMPLE_AWI_CONTEXT = {
    "league_rank": 3,
    "total_players": 50,
    "position_avg": 9.8,
    "cross_half_r": 0.72,
}

SAMPLE_PQI_CONTEXT = {
    "mean_pqi": 74.3,
    "orientation_mean": 68.1,
    "stance_mean": 79.5,
    "proximity_mean": 75.2,
}

SAMPLE_MATCH_CONTEXT = {
    "match_label": "Bayern vs Dortmund",
    "opponent": "Dortmund",
}


def _make_throttle_error() -> ClientError:
    return ClientError(
        {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}},
        "InvokeModel",
    )


def _make_mock_response(text: str = "Test narrative") -> MagicMock:
    """Build a mock boto3 response whose body.read() returns valid JSON."""
    body_bytes = json.dumps({"content": [{"text": text}]}).encode()
    mock_response = MagicMock()
    mock_response["body"].read.return_value = body_bytes
    return mock_response


# ---------------------------------------------------------------------------
# build_player_prompt tests
# ---------------------------------------------------------------------------


def test_build_player_prompt_contains_required_fields():
    prompt = build_player_prompt(
        SAMPLE_PLAYER_ROW,
        SAMPLE_AWI_CONTEXT,
        SAMPLE_PQI_CONTEXT,
        SAMPLE_MATCH_CONTEXT,
    )
    assert "Thomas Müller" in prompt
    assert "AM" in prompt
    assert "12.5" in prompt          # AWI value
    assert "74.3" in prompt          # mean_pqi (PQI value)
    assert "68.1" in prompt          # orientation_mean
    assert "79.5" in prompt          # stance_mean
    assert "75.2" in prompt          # proximity_mean


def test_build_player_prompt_length():
    prompt = build_player_prompt(
        SAMPLE_PLAYER_ROW,
        SAMPLE_AWI_CONTEXT,
        SAMPLE_PQI_CONTEXT,
        SAMPLE_MATCH_CONTEXT,
    )
    assert len(prompt) < 4000 * 4


# ---------------------------------------------------------------------------
# generate_player_narrative tests
# ---------------------------------------------------------------------------


def test_generate_player_narrative_success():
    mock_client = MagicMock()
    body_bytes = json.dumps({"content": [{"text": "Test narrative"}]}).encode()
    mock_body = MagicMock()
    mock_body.read.return_value = body_bytes
    mock_client.invoke_model.return_value = {"body": mock_body}

    result = generate_player_narrative(
        mock_client,
        SAMPLE_PLAYER_ROW,
        SAMPLE_AWI_CONTEXT,
        SAMPLE_PQI_CONTEXT,
        SAMPLE_MATCH_CONTEXT,
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
    )
    assert isinstance(result, str)
    assert len(result) > 0


@patch("time.sleep", return_value=None)
def test_generate_player_narrative_throttling_retry(mock_sleep):
    mock_client = MagicMock()

    body_bytes = json.dumps({"content": [{"text": "Test narrative"}]}).encode()
    mock_body = MagicMock()
    mock_body.read.return_value = body_bytes
    success_response = {"body": mock_body}

    mock_client.invoke_model.side_effect = [
        _make_throttle_error(),
        _make_throttle_error(),
        success_response,
    ]

    result = generate_player_narrative(
        mock_client,
        SAMPLE_PLAYER_ROW,
        SAMPLE_AWI_CONTEXT,
        SAMPLE_PQI_CONTEXT,
        SAMPLE_MATCH_CONTEXT,
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
    )
    assert result == "Test narrative"
    assert mock_client.invoke_model.call_count == 3


@patch("time.sleep", return_value=None)
def test_generate_player_narrative_nova_format(mock_sleep):
    """Nova models use a different response format — verify parsing works."""
    mock_client = MagicMock()
    body_bytes = json.dumps({
        "output": {"message": {"content": [{"text": "Nova narrative"}]}}
    }).encode()
    mock_body = MagicMock()
    mock_body.read.return_value = body_bytes
    mock_client.invoke_model.return_value = {"body": mock_body}

    result = generate_player_narrative(
        mock_client,
        SAMPLE_PLAYER_ROW,
        SAMPLE_AWI_CONTEXT,
        SAMPLE_PQI_CONTEXT,
        SAMPLE_MATCH_CONTEXT,
        model_id="eu.amazon.nova-lite-v1:0",
    )
    assert result == "Nova narrative"


@patch("time.sleep", return_value=None)
def test_generate_player_narrative_all_retries_fail(mock_sleep):
    mock_client = MagicMock()
    mock_client.invoke_model.side_effect = _make_throttle_error()

    with pytest.raises(BedrockClientError):
        generate_player_narrative(
            mock_client,
            SAMPLE_PLAYER_ROW,
            SAMPLE_AWI_CONTEXT,
            SAMPLE_PQI_CONTEXT,
            SAMPLE_MATCH_CONTEXT,
        )

    assert mock_client.invoke_model.call_count == 3


# ---------------------------------------------------------------------------
# batch_generate_narratives tests
# ---------------------------------------------------------------------------


def _make_synthetic_dfs(n: int = 3):
    """Create minimal synthetic awi_df and pqi_df with n players."""
    awi_df = pd.DataFrame({
        "jersey": list(range(1, n + 1)),
        "team": ["TeamA"] * n,
        "match_id": ["match_001"] * n,
        "phase_label": ["build_up"] * n,
        "awi_per_minute": [10.0 + i for i in range(n)],
        "position": ["CM"] * n,
        "name": [f"Player {i}" for i in range(1, n + 1)],
    })
    pqi_df = pd.DataFrame({
        "jersey": list(range(1, n + 1)),
        "team": ["TeamA"] * n,
        "match_id": ["match_001"] * n,
        "phase_label": ["build_up"] * n,
        "mean_pqi": [70.0 + i for i in range(n)],
        "orientation_mean": [65.0] * n,
        "stance_mean": [75.0] * n,
        "proximity_mean": [72.0] * n,
    })
    return awi_df, pqi_df


def test_batch_generate_narratives_returns_dataframe():
    awi_df, pqi_df = _make_synthetic_dfs(3)
    mock_client = MagicMock()

    with patch("src.bedrock_client.generate_player_narrative", return_value="Mock narrative"):
        result = batch_generate_narratives(mock_client, awi_df, pqi_df, top_n=2)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["jersey", "team", "match_id", "phase_label", "narrative"]
    assert len(result) == 2


def test_batch_generate_narratives_skips_failures():
    awi_df, pqi_df = _make_synthetic_dfs(3)
    mock_client = MagicMock()

    call_count = {"n": 0}

    def side_effect(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise BedrockClientError("Simulated failure for first player")
        return "Mock narrative"

    with patch("src.bedrock_client.generate_player_narrative", side_effect=side_effect):
        result = batch_generate_narratives(mock_client, awi_df, pqi_df, top_n=3)

    # First player failed, remaining 2 should succeed
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert all(result["narrative"] == "Mock narrative")
