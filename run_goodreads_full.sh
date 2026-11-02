#!/bin/bash
# HARP - Goodreads (11734 users, full test split)
#
# Usage:  DATA_DIR=/path/to/instructrec-goodreads bash run_goodreads_full.sh
#         (DATA_DIR defaults to dataset.data_dir in configs/com_goodreads_full.yaml;
#          extra arguments are passed on, e.g. --n_eval_users 100)

set -e
cd "$(dirname "$0")"

TIMESTAMP=$(date '+%m%d_%H%M')
RUN_TAG="harp_goodreads_full_${TIMESTAMP}"
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${RUN_TAG}.log"

# Core components
export HARP_HOMOPHILY=1           # user-user homophily gate + profile update in Stage-W
export HARP_ITEM_PROFILE=1        # item profile (zero LLM calls)
export HARP_USER_PROFILE=1        # structured user profile
export HARP_ENRICH_CANDIDATES=1   # candidate metadata in the reranker prompt
export HARP_RERANKER_PROFILE=1    # user profile injected into the reranker prompt


python3 scripts/run_com_train.py \
  --config configs/com_goodreads_full.yaml \
  ${DATA_DIR:+--data_dir "$DATA_DIR"} \
  --n_eval_candidates 10 \
  --seed 42 \
  --parallel_workers 8 \
  --warmup_parallel_workers 8 \
  --warmup_rounds 1 \
  --output_dir results \
  --run_tag "${RUN_TAG}" \
  "$@" \
  2>&1 | tee "${LOG_FILE}"

echo "Results: results/${RUN_TAG}/"
