#!/bin/bash
# scripts/refresh_streamlit_secrets.sh
# Refreshes SSO credentials and outputs them in Streamlit secrets TOML format.
#
# Usage:
#   bash scripts/refresh_streamlit_secrets.sh
#
# Then copy-paste the output into Streamlit Cloud > App settings > Secrets.

set -euo pipefail

PROFILE="${AWS_PROFILE:-slalom_IsbUsersPS-603974305500}"

echo "==> Logging in to AWS SSO (opens browser)..."
aws sso login --profile "$PROFILE"

echo ""
echo "==> Exporting credentials..."
CREDS=$(aws configure export-credentials --profile "$PROFILE" --format env)

ACCESS_KEY=$(echo "$CREDS" | grep AWS_ACCESS_KEY_ID | cut -d= -f2)
SECRET_KEY=$(echo "$CREDS" | grep AWS_SECRET_ACCESS_KEY | cut -d= -f2)
SESSION_TOKEN=$(echo "$CREDS" | grep AWS_SESSION_TOKEN | cut -d= -f2)
EXPIRATION=$(echo "$CREDS" | grep AWS_CREDENTIAL_EXPIRATION | cut -d= -f2)

echo ""
echo "============================================================"
echo "  STREAMLIT SECRETS (copy everything below into App Settings)"
echo "  Expires: $EXPIRATION"
echo "============================================================"
echo ""
echo "[aws]"
echo "bucket = \"hackathon-data-603974305500\""
echo "results_prefix = \"results\""
echo "aws_access_key_id = \"$ACCESS_KEY\""
echo "aws_secret_access_key = \"$SECRET_KEY\""
echo "aws_session_token = \"$SESSION_TOKEN\""
echo "region_name = \"eu-central-1\""
echo ""
echo "============================================================"
