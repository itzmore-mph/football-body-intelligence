"""
bedrock_client.py

Generates natural language player reports using Amazon Bedrock Claude Sonnet 4.6.
No hardcoded credentials - all AWS access via boto3.Session(profile_name=...).
"""

import json
import logging
import os
import time

import boto3
import pandas as pd
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "eu.amazon.nova-lite-v1:0")
BEDROCK_REGION = os.environ.get("BEDROCK_REGION", "eu-central-1")

# Models that use the Amazon Nova converse API format (not Anthropic messages format)
_NOVA_MODEL_PREFIXES = ("eu.amazon.nova", "us.amazon.nova", "amazon.nova")


class BedrockClientError(Exception):
    """Raised after exhausting all retry attempts on Bedrock API calls."""


def create_bedrock_client(
    region: str = BEDROCK_REGION,
    profile_name: str | None = None,
) -> "boto3.client":
    """Create a boto3 bedrock-runtime client.

    Uses the same AWS session pattern as eda_helpers.create_session().
    Credentials come from the environment (AWS_PROFILE env var) or the
    explicit profile_name argument - never hardcoded.

    Args:
        region:       AWS region for Bedrock (default: eu-central-1).
        profile_name: Optional AWS profile name. Falls back to AWS_PROFILE
                      env var if not provided.

    Returns:
        boto3 bedrock-runtime client.
    """
    profile = profile_name or os.environ.get("AWS_PROFILE")
    session = boto3.Session(profile_name=profile, region_name=region)
    return session.client("bedrock-runtime", region_name=region)


def build_player_prompt(
    player_row: dict,
    awi_context: dict,
    pqi_context: dict,
    match_context: dict,
) -> str:
    """Construct a structured prompt for player narrative generation.

    Args:
        player_row:    Dict with keys: name, position, match_id, phase_label,
                       awi_per_minute.
        awi_context:   Dict with keys: league_rank, total_players, position_avg,
                       cross_half_r (optional).
        pqi_context:   Dict with keys: mean_pqi, orientation_mean, stance_mean,
                       proximity_mean.
        match_context: Dict with keys: match_label, opponent.

    Returns:
        Prompt string < 4000 tokens suitable for Claude Sonnet 4.6.
    """
    cross_half = awi_context.get("cross_half_r", "N/A")
    cross_half_str = f"{cross_half:.3f}" if isinstance(cross_half, float) else str(cross_half)

    return (
        f"You are a football analytics expert writing a concise player intelligence report.\n\n"
        f"Player: {player_row['name']} ({player_row['position']})\n"
        f"Match: {match_context['match_label']} - {player_row['phase_label']}\n\n"
        f"AWARENESS INDEX (AWI):\n"
        f"- AWI: {player_row['awi_per_minute']:.1f} scans/min\n"
        f"- League rank: #{awi_context['league_rank']} of {awi_context['total_players']}\n"
        f"- Position average: {awi_context['position_avg']:.1f} scans/min\n"
        f"- Cross-half consistency R: {cross_half_str}\n\n"
        f"PRESSURE QUALITY INDEX (PQI):\n"
        f"- Mean PQI: {pqi_context['mean_pqi']:.1f}/100\n"
        f"- Orientation score: {pqi_context['orientation_mean']:.1f}/100\n"
        f"- Stance score: {pqi_context['stance_mean']:.1f}/100\n"
        f"- Proximity score: {pqi_context['proximity_mean']:.1f}/100\n\n"
        f"Write a 2-paragraph scouting report (max 130 words). Rules:\n"
        f"- Open each claim with a specific number from the data above, not an adjective.\n"
        f"- Tie AWI to this player's POSITION and the PHASE shown "
        f"(e.g. a deep midfielder scanning before progression, a drop in the 2nd half).\n"
        f"- Name the single weakest PQI sub-score and the mechanical reason it lowers the press.\n"
        f"- End with one concrete recommendation a club can act on in recruitment OR training.\n"
        f"- Banned: 'formidable', 'exceptional', 'world-class', 'presence', and any praise "
        f"not backed by a number. No generic intros."
    )


def generate_player_narrative(
    client,
    player_row: dict,
    awi_context: dict,
    pqi_context: dict,
    match_context: dict,
    model_id: str = BEDROCK_MODEL_ID,
    max_tokens: int = 512,
) -> str:
    """Generate a player narrative via AWS Bedrock Claude Sonnet 4.6.

    Args:
        client:        boto3 bedrock-runtime client.
        player_row:    Dict with player KPI fields.
        awi_context:   Dict with AWI context (league_rank, position_avg, etc.).
        pqi_context:   Dict with PQI context (mean_pqi, sub-score means).
        match_context: Dict with match_label, opponent.
        model_id:      Bedrock model ID (default: anthropic.claude-sonnet-4-6).
        max_tokens:    Maximum tokens in the response.

    Returns:
        Non-empty markdown narrative string.

    Raises:
        BedrockClientError: After 3 failed retry attempts.
    """
    prompt = build_player_prompt(player_row, awi_context, pqi_context, match_context)
    return _invoke_with_retry(client, prompt, model_id=model_id, max_tokens=max_tokens)


def _invoke_with_retry(
    client,
    prompt: str,
    model_id: str = BEDROCK_MODEL_ID,
    max_tokens: int = 512,
    max_retries: int = 3,
) -> str:
    """Call Bedrock with exponential backoff on ThrottlingException.

    Args:
        client:      boto3 bedrock-runtime client.
        prompt:      Prompt string to send.
        model_id:    Bedrock model ID.
        max_tokens:  Maximum tokens in the response.
        max_retries: Maximum number of retry attempts (default: 3).

    Returns:
        Response text string.

    Raises:
        BedrockClientError: After exhausting all retries.
    """
    is_nova = model_id.startswith(_NOVA_MODEL_PREFIXES)

    if is_nova:
        body = json.dumps({
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxTokens": max_tokens},
        })
    else:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        })

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.invoke_model(
                modelId=model_id,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            result = json.loads(response["body"].read())
            if is_nova:
                return result["output"]["message"]["content"][0]["text"]
            return result["content"][0]["text"]
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code == "ThrottlingException":
                wait = 2 ** (attempt + 1)  # 2s, 4s, 8s
                logger.warning(
                    "Bedrock ThrottlingException (attempt %d/%d). Waiting %ds...",
                    attempt + 1, max_retries, wait,
                )
                time.sleep(wait)
                last_exc = e
            else:
                raise BedrockClientError(
                    f"Bedrock API error ({code}): {e.response['Error']['Message']}"
                ) from e
        except Exception as e:
            last_exc = e
            wait = 2 ** (attempt + 1)
            logger.warning(
                "Bedrock call failed (attempt %d/%d): %s. Waiting %ds...",
                attempt + 1, max_retries, e, wait,
            )
            time.sleep(wait)

    raise BedrockClientError(
        f"Bedrock call failed after {max_retries} attempts. Last error: {last_exc}"
    )


def batch_generate_narratives(
    client,
    awi_df: pd.DataFrame,
    pqi_df: pd.DataFrame,
    top_n: int = 10,
    model_id: str = BEDROCK_MODEL_ID,
) -> pd.DataFrame:
    """Generate narratives for the top-N players ranked by awi_per_minute.

    Merges awi_df and pqi_df on [jersey, team, match_id, phase_label], ranks
    players by awi_per_minute descending, and calls generate_player_narrative
    for each of the top_n players. Single-player failures are logged and skipped.

    Args:
        client:   boto3 bedrock-runtime client.
        awi_df:   DataFrame with AWI results (must include jersey, team, match_id,
                  phase_label, awi_per_minute, position columns).
        pqi_df:   DataFrame with PQI results (must include jersey, team, match_id,
                  phase_label, mean_pqi, orientation_mean, stance_mean,
                  proximity_mean columns).
        top_n:    Number of top players to generate narratives for (default: 10).
        model_id: Bedrock model ID.

    Returns:
        DataFrame with columns [jersey, team, match_id, phase_label, narrative].
    """
    merge_keys = ["jersey", "team", "match_id", "phase_label"]
    # suffixes=('', '_pqi') keeps AWI columns (name, position, coverage_pct)
    # under their original names; duplicate PQI columns get a _pqi suffix.
    combined = awi_df.merge(pqi_df, on=merge_keys, how="inner", suffixes=("", "_pqi"))

    # Rank by awi_per_minute descending; rank is 1-based
    combined = combined.copy()
    combined["_rank"] = combined["awi_per_minute"].rank(ascending=False, method="first").astype(int)
    total_players = len(combined)

    # Compute position averages for awi_context
    position_avgs = combined.groupby("position")["awi_per_minute"].mean()

    top_players = combined.nsmallest(top_n, "_rank")

    records = []
    for _, row in top_players.iterrows():
        try:
            player_row = {
                "name": row.get("name", f"Player {row['jersey']}"),
                "position": row.get("position", "Unknown"),
                "match_id": row["match_id"],
                "phase_label": row["phase_label"],
                "awi_per_minute": row["awi_per_minute"],
            }
            awi_context = {
                "league_rank": int(row["_rank"]),
                "total_players": total_players,
                "position_avg": position_avgs.get(row.get("position", "Unknown"), 0.0),
            }
            pqi_context = {
                "mean_pqi": row.get("mean_pqi", 0.0),
                "orientation_mean": row.get("orientation_mean", 0.0),
                "stance_mean": row.get("stance_mean", 0.0),
                "proximity_mean": row.get("proximity_mean", 0.0),
            }
            match_context = {
                "match_label": row["match_id"],
                "opponent": "",
            }
            narrative = generate_player_narrative(
                client, player_row, awi_context, pqi_context, match_context,
                model_id=model_id,
            )
            records.append({
                "jersey": row["jersey"],
                "team": row["team"],
                "match_id": row["match_id"],
                "phase_label": row["phase_label"],
                "narrative": narrative,
            })
        except Exception as exc:
            logger.warning(
                "Failed to generate narrative for jersey=%s team=%s match=%s phase=%s: %s",
                row.get("jersey"), row.get("team"), row.get("match_id"),
                row.get("phase_label"), exc,
            )

    return pd.DataFrame(records, columns=["jersey", "team", "match_id", "phase_label", "narrative"])


def generate_match_summary(
    client,
    match_id: str,
    awi_df: pd.DataFrame,
    pqi_df: pd.DataFrame,
    model_id: str = BEDROCK_MODEL_ID,
    max_tokens: int = 1024,
) -> str:
    """Generate a match-level narrative summary via AWS Bedrock.

    Filters both DataFrames to the given match_id, identifies top AWI and PQI
    performers, and builds a structured prompt covering tactical patterns.

    Args:
        client:     boto3 bedrock-runtime client.
        match_id:   Match identifier to summarise.
        awi_df:     DataFrame with AWI results.
        pqi_df:     DataFrame with PQI results.
        model_id:   Bedrock model ID.
        max_tokens: Maximum tokens in the response (default: 1024).

    Returns:
        Markdown string with the match summary narrative.

    Raises:
        BedrockClientError: After exhausting all retry attempts.
    """
    match_awi = awi_df[awi_df["match_id"] == match_id].copy()
    match_pqi = pqi_df[pqi_df["match_id"] == match_id].copy()

    # Top AWI performers
    top_awi = match_awi.nlargest(5, "awi_per_minute") if not match_awi.empty else match_awi
    top_awi_lines = []
    for _, row in top_awi.iterrows():
        name = row.get("name", f"Player {row.get('jersey', '?')}")
        pos = row.get("position", "Unknown")
        awi = row.get("awi_per_minute", 0.0)
        top_awi_lines.append(f"  - {name} ({pos}): {awi:.1f} scans/min")

    # Top PQI performers
    top_pqi = match_pqi.nlargest(5, "mean_pqi") if not match_pqi.empty else match_pqi
    top_pqi_lines = []
    for _, row in top_pqi.iterrows():
        name = row.get("name", f"Player {row.get('jersey', '?')}")
        pos = row.get("position", "Unknown")
        pqi = row.get("mean_pqi", 0.0)
        top_pqi_lines.append(f"  - {name} ({pos}): {pqi:.1f}/100")

    # Tactical pattern summary
    avg_awi = match_awi["awi_per_minute"].mean() if not match_awi.empty else 0.0
    avg_pqi = match_pqi["mean_pqi"].mean() if not match_pqi.empty else 0.0

    awi_block = "\n".join(top_awi_lines) if top_awi_lines else "  No AWI data available."
    pqi_block = "\n".join(top_pqi_lines) if top_pqi_lines else "  No PQI data available."

    prompt = (
        f"You are a football analytics expert writing a match intelligence summary.\n\n"
        f"Match ID: {match_id}\n\n"
        f"AWARENESS INDEX (AWI) - Top Performers:\n{awi_block}\n"
        f"Match average AWI: {avg_awi:.1f} scans/min\n\n"
        f"PRESSURE QUALITY INDEX (PQI) - Top Performers:\n{pqi_block}\n"
        f"Match average PQI: {avg_pqi:.1f}/100\n\n"
        f"Write a 3-paragraph match intelligence summary (max 250 words) covering:\n"
        f"1. Cognitive awareness patterns - which players or positions showed elite scanning behaviour\n"
        f"2. Pressing mechanics - which players demonstrated the best body mechanics under pressure\n"
        f"3. Tactical insights - what the combined AWI+PQI data reveals about team pressing strategy\n"
        f"Format the output as markdown with clear section headers."
    )

    return _invoke_with_retry(client, prompt, model_id=model_id, max_tokens=max_tokens)
