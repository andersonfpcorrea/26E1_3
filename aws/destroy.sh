#!/usr/bin/env bash
# destroy.sh — Remove all AWS infrastructure.
# After running, monthly cost = $0.00.
#
# Usage:
#   cd aws
#   ./destroy.sh <AWS_PROFILE>
set -euo pipefail

PROFILE="${1:?Usage: ./destroy.sh <AWS_PROFILE>}"
export AWS_PROFILE="$PROFILE"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Cardio ML — Destroying AWS stack ==="
echo "Profile: $AWS_PROFILE"
echo ""

cd "$SCRIPT_DIR"
npx cdk destroy --force --profile "$PROFILE"

rm -rf "$SCRIPT_DIR/src/lambda/model/pipeline.joblib"

echo ""
echo "=== Stack destroyed. Future cost: \$0.00 ==="
