#!/bin/sh
# Zip the code (no venv, results or notebooks) for upload as a PRIVATE Kaggle dataset.
set -e
cd "$(dirname "$0")/.."
rm -f kaggle/grpo_query_code.zip
zip -rq kaggle/grpo_query_code.zip data retrieval rewards train eval configs tests pytest.ini \
  -x '*/__pycache__/*' '*.pyc'
echo "wrote kaggle/grpo_query_code.zip ($(du -h kaggle/grpo_query_code.zip | cut -f1))"
