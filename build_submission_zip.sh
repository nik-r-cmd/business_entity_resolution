#!/bin/bash
# Usage: ./build_submission_zip.sh <team_name>
# Run from inside the repo root, with output/matching_results.tsv and output/candidate_pairs.tsv already generated.
set -e
TEAM="${1:?Usage: ./build_submission_zip.sh <team_name>}"
rm -rf /tmp/pkg && mkdir -p /tmp/pkg/code/business_entity_resolution /tmp/pkg/output

rsync -a --exclude '.git' --exclude 'dataset' --exclude 'artifacts' --exclude 'output' \
      --exclude 'experiments' --exclude 'submissions' --exclude 'tests' --exclude '__pycache__' \
      --exclude '*.log' \
      ./ /tmp/pkg/code/business_entity_resolution/

cp output/matching_results.tsv output/candidate_pairs.tsv /tmp/pkg/output/
cp Documentation_template.md /tmp/pkg/

cd /tmp/pkg && zip -r ~/"${TEAM}_submission.zip" .
echo "Built: ~/${TEAM}_submission.zip"
unzip -l ~/"${TEAM}_submission.zip"
