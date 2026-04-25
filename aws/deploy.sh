#!/usr/bin/env bash
# deploy.sh — Export model, install CDK deps and deploy the stack.
#
# Usage:
#   cd aws
#   ./deploy.sh <AWS_PROFILE>
#
# Example:
#   ./deploy.sh my-personal-account
#
# Prerequisites:
#   - AWS CLI configured with the profile
#   - CDK CLI installed (npm install -g aws-cdk)
#   - Docker running
#   - Model champion registered in MLflow (make select)
set -euo pipefail

PROFILE="${1:?Usage: ./deploy.sh <AWS_PROFILE>}"
export AWS_PROFILE="$PROFILE"
export AWS_PAGER=""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_DIR="$SCRIPT_DIR/src/lambda/model"

echo "=== Cardio ML — Deploy to AWS ==="
echo ""

# Verify AWS credentials
IDENTITY=$(aws sts get-caller-identity --profile "$PROFILE" --output json 2>&1) || {
    echo "Credentials for profile '$PROFILE' not found or expired."
    echo ""
    echo "Please log in first:"
    echo "  aws sso login --profile $PROFILE"
    echo ""
    echo "Or configure the profile:"
    echo "  aws configure --profile $PROFILE"
    exit 1
}

ACCOUNT_ID=$(echo "$IDENTITY" | grep -o '"Account": "[^"]*"' | cut -d'"' -f4)
ARN=$(echo "$IDENTITY" | grep -o '"Arn": "[^"]*"' | cut -d'"' -f4)

echo "Profile:  $PROFILE"
echo "Account:  $ACCOUNT_ID"
echo "Identity: $ARN"
echo "Region:   us-east-1"
echo "Project:  $PROJECT_ROOT"
echo ""
read -rp "Proceed with deploy? [y/N] " confirm
if [[ ! "$confirm" =~ ^[yY]$ ]]; then
    echo "Aborted."
    exit 0
fi
echo ""

# ------------------------------------------------------------------
# 1. Export model from MLflow to joblib
# ------------------------------------------------------------------
echo "[1/4] Exporting model from MLflow..."
mkdir -p "$MODEL_DIR"

cd "$PROJECT_ROOT"
uv run python -c "
import os, sys, joblib, mlflow
from cardio_ml.config import MLFLOW_TRACKING_URI, MLFLOW_REGISTERED_MODEL_NAME

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = mlflow.tracking.MlflowClient()

try:
    versions = client.get_latest_versions(MLFLOW_REGISTERED_MODEL_NAME, stages=['Production'])
except Exception:
    versions = []

if not versions:
    all_v = client.search_model_versions(f\"name='{MLFLOW_REGISTERED_MODEL_NAME}'\")
    if not all_v:
        print('ERROR: No registered model. Run make select first.')
        sys.exit(1)
    version = max(all_v, key=lambda v: int(v.version))
else:
    version = versions[0]

model_uri = f'models:/{MLFLOW_REGISTERED_MODEL_NAME}/{version.version}'
pipeline = mlflow.sklearn.load_model(model_uri)

output = '$MODEL_DIR/pipeline.joblib'
joblib.dump(pipeline, output)
print(f'  Model v{version.version} (run {version.run_id[:8]}) -> {output}')
print(f'  Size: {os.path.getsize(output) / 1024 / 1024:.1f} MB')
"

if [ ! -f "$MODEL_DIR/pipeline.joblib" ]; then
    echo "ERROR: Failed to export model."
    exit 1
fi

# ------------------------------------------------------------------
# 2. Install CDK dependencies
# ------------------------------------------------------------------
echo ""
echo "[2/4] Installing CDK dependencies..."
cd "$SCRIPT_DIR"
npm install --quiet

# ------------------------------------------------------------------
# 3. CDK bootstrap (first time per account/region)
# ------------------------------------------------------------------
echo ""
echo "[3/4] CDK bootstrap..."
npx cdk bootstrap "aws://$ACCOUNT_ID/us-east-1" --profile "$PROFILE" 2>/dev/null || true

# ------------------------------------------------------------------
# 4. Deploy
# ------------------------------------------------------------------
echo ""
echo "[4/4] Deploying stack..."
npx cdk deploy --require-approval never --profile "$PROFILE"

# ------------------------------------------------------------------
# 5. Warm up the Lambda (trigger a cold start now, not on first user request)
# ------------------------------------------------------------------
echo ""
echo "[5/5] Warming up Lambda..."
API_URL=$(aws cloudformation describe-stacks \
    --stack-name CardioMlStack \
    --query "Stacks[0].Outputs[?OutputKey=='ApiUrl'].OutputValue" \
    --output text --profile "$PROFILE")

if [ -n "$API_URL" ]; then
    # First call triggers cold start — wait for it
    curl -sS --max-time 30 "${API_URL}health" > /dev/null 2>&1 || true
    # Second call confirms it's warm
    HEALTH=$(curl -sS --max-time 10 "${API_URL}health" 2>/dev/null)
    echo "  $HEALTH"
    echo "  API ready at: ${API_URL}"
fi

echo ""
echo "=== Deploy complete ==="
echo ""
echo "To destroy after grading:"
echo "  cd aws && ./destroy.sh $PROFILE"
