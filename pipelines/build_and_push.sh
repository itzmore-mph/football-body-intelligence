#!/usr/bin/env bash
# pipelines/build_and_push.sh
# Builds the Docker image and pushes it to Amazon ECR.
#
# Prerequisites:
#   - Docker running locally
#   - AWS CLI configured (SSO or env vars)
#   - ECR repository exists (created once, see Step 1 below)
#
# Usage:
#   ./pipelines/build_and_push.sh                     # builds + pushes with tag=latest
#   IMAGE_TAG=v1.2 ./pipelines/build_and_push.sh      # custom tag
#
# The script must be run from the repo root (where Dockerfile lives).

set -euo pipefail

# ── Configuration (override via env vars) ────────────────────────────────────
REGION="${AWS_DEFAULT_REGION:-eu-central-1}"
PROFILE="${AWS_PROFILE:-}"
ECR_REPO="${ECR_REPO:-football-bi-processing}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# Build --profile flag only if AWS_PROFILE is set (supports both SSO and default chain)
AWS_PROFILE_FLAG=""
if [ -n "${PROFILE}" ]; then
  AWS_PROFILE_FLAG="--profile ${PROFILE}"
fi

ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity ${AWS_PROFILE_FLAG} --query Account --output text)}"

REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
FULL_IMAGE="${REGISTRY}/${ECR_REPO}:${IMAGE_TAG}"

echo "=================================================="
echo "  Building: ${FULL_IMAGE}"
echo "  Region:   ${REGION}"
echo "=================================================="

# ── Step 1: Create ECR repo if it doesn't exist (idempotent) ─────────────────
echo ""
echo "Ensuring ECR repository exists..."
aws ecr describe-repositories \
    --repository-names "${ECR_REPO}" \
    --region "${REGION}" ${AWS_PROFILE_FLAG} > /dev/null 2>&1 || \
aws ecr create-repository \
    --repository-name "${ECR_REPO}" \
    --region "${REGION}" ${AWS_PROFILE_FLAG} \
    --image-scanning-configuration scanOnPush=true \
    --encryption-configuration encryptionType=AES256
echo "  Repository: ${REGISTRY}/${ECR_REPO}"

# ── Step 2: Authenticate Docker with ECR ─────────────────────────────────────
echo ""
echo "Authenticating with ECR..."
aws ecr get-login-password --region "${REGION}" ${AWS_PROFILE_FLAG} \
    | docker login --username AWS --password-stdin "${REGISTRY}"

# ── Step 3: Build image from repo root ───────────────────────────────────────
echo ""
echo "Building Docker image (this may take a few minutes on first run)..."
docker build \
    --platform linux/amd64 \
    --tag "${FULL_IMAGE}" \
    --tag "${REGISTRY}/${ECR_REPO}:latest" \
    -f Dockerfile \
    .

# ── Step 4: Push to ECR ──────────────────────────────────────────────────────
echo ""
echo "Pushing to ECR..."
docker push "${FULL_IMAGE}"
# Also push/update the 'latest' tag (unless IMAGE_TAG is already 'latest')
if [ "${IMAGE_TAG}" != "latest" ]; then
    docker push "${REGISTRY}/${ECR_REPO}:latest"
fi

echo ""
echo "=================================================="
echo "  Done!"
echo "  IMAGE_URI=${FULL_IMAGE}"
echo ""
echo "  Next step — run the pipeline:"
echo "    export SM_IMAGE_URI=${FULL_IMAGE}"
echo "    export S3_BUCKET=<your-bucket>"
echo "    export CHALLENGE_PREFIX=<your-prefix>"
echo "    export SM_ROLE_ARN=arn:aws:iam::${ACCOUNT_ID}:role/SageMakerExecutionRole"
echo "    python pipelines/sagemaker_pipeline.py --action run"
echo "=================================================="
