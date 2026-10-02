#!/usr/bin/env bash
# Code + General in one run on uni-agent. Wraps recipes/code/run_train.sh with the mixed config.
#
#   CODE_TRAIN_DATA=/data/code.parquet GENERAL_TRAIN_DATA=$GA_TASK_ROOT/parquet/train.parquet \
#   GA_TASK_ROOT=... GA_JUDGE_URL=... GA_JUDGE_MODEL=... GA_JUDGE_KEY_FILE=... \
#   MODEL_PATH=... VAL_DATA=... bash recipes/mixed/run_mixed.sh [hydra overrides]
#
# The Sample Mixer (per-source quotas) is a separate step; until it lands the dataloader draws
# from the concatenated parquets, so each batch's source mix follows the data sizes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${CODE_TRAIN_DATA:?set CODE_TRAIN_DATA to the Code training parquet}"
: "${GENERAL_TRAIN_DATA:?set GENERAL_TRAIN_DATA to the General training parquet}"
if [ "${SANDBOX:-kubernetes}" = "docker" ]; then
  # General tasks run a main container plus a sidecar with the task's MCP servers; only the
  # Kubernetes sidecar environment exists so far.
  echo "SANDBOX=docker covers Code tasks only (recipes/code/run_train.sh); General needs the Kubernetes sidecar environment" >&2
  exit 1
fi
: "${GA_TASK_ROOT:?set GA_TASK_ROOT to the open_source_env bundle root (General task dirs)}"
: "${GA_JUDGE_URL:?set GA_JUDGE_URL to the General rubric judge (OpenAI-compatible base URL)}"
if [ ! -d "${GA_TASK_ROOT}/envs" ]; then
  echo "GA_TASK_ROOT has no envs/ directory: ${GA_TASK_ROOT}" >&2
  exit 1
fi
if [ -z "${GA_JUDGE_KEY_FILE:-}" ] && [ -z "${GA_JUDGE_KEY:-}" ]; then
  echo "set GA_JUDGE_KEY_FILE (a key file readable on every node; recommended) or GA_JUDGE_KEY" >&2
  exit 1
fi
if [ -n "${GA_JUDGE_KEY_FILE:-}" ] && [ ! -r "${GA_JUDGE_KEY_FILE}" ]; then
  echo "GA_JUDGE_KEY_FILE is not readable here: ${GA_JUDGE_KEY_FILE} (it must be readable on every node)" >&2
  exit 1
fi

export TRAIN_DATA="${CODE_TRAIN_DATA},${GENERAL_TRAIN_DATA}"
export CONFIG_PATH="${SCRIPT_DIR}/config"
export CONFIG_NAME=mixed

# General's environment and judge read these on the workers (Ray actors get the raylet's
# environment, not this shell's).
GENERAL_ENV=(
  +ray_kwargs.ray_init.runtime_env.env_vars.GA_TASK_ROOT="${GA_TASK_ROOT}"
  +ray_kwargs.ray_init.runtime_env.env_vars.GA_JUDGE_URL="'${GA_JUDGE_URL}'"
  +ray_kwargs.ray_init.runtime_env.env_vars.GA_JUDGE_MODEL="'${GA_JUDGE_MODEL:-gpt-4o-mini}'"
  +ray_kwargs.ray_init.runtime_env.env_vars.GA_JUDGE_API="'${GA_JUDGE_API:-chat}'"
  +ray_kwargs.ray_init.runtime_env.env_vars.GA_JUDGE_KEY_FILE="'${GA_JUDGE_KEY_FILE:-}'"
  +ray_kwargs.ray_init.runtime_env.env_vars.DOCKER_REGISTRY="'${DOCKER_REGISTRY:-}'"
)
if [ -n "${GA_JUDGE_KEY:-}" ]; then
  echo "warning: GA_JUDGE_KEY goes into the resolved config and run manifest; prefer GA_JUDGE_KEY_FILE" >&2
  GENERAL_ENV+=(+ray_kwargs.ray_init.runtime_env.env_vars.GA_JUDGE_KEY="'${GA_JUDGE_KEY}'")
fi

exec bash "${SCRIPT_DIR}/../code/run_train.sh" "${GENERAL_ENV[@]}" "$@"
