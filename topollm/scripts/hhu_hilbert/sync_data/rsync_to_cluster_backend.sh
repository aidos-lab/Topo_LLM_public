#!/bin/bash

# Unified local -> HHU Hilbert rsync helper
#
# Usage:
#   rsync_to_hhu_hilbert.sh --group <name> [--group <name> ...] [--dry-run]
#   rsync_to_hhu_hilbert.sh --list
#
# Groups:
#   git-repo          Sync local repo (with excludes) to /gpfs/project/<user>/git-source/Topo_LLM
#   hf-cache          Sync local Hugging Face cache to /gpfs/project/<user>/models
#   nltk-data         Sync local $HOME/nltk_data to /home/<user>/nltk_data
#   trippy-data       Sync post_processed_cached_features for Trippy & Trippy-R
#   trippy-r-code     Sync Trippy-R helper scripts and code folder
#   emoloop-models    Sync local data/models/EmoLoop directory
#
# Implementation:
#   - Data-driven: each selected group contributes one or more (src, dest, flags) pairs.
#   - One unified rsync loop executes all pairs with consistent logging and dry-run support.
#
# Notes:
# - Requires environment variables from .env in repo root; this script will source it.
# - Supports multiple --group flags in a single run.
# - Adds --dry-run pass-through for all rsync calls.

set -o pipefail

# ------------------------------
# Helpers
# ------------------------------

usage() {
  echo "Usage: $0 --group <name> [--group <name> ...] [--dry-run] | --list"
  exit 1
}

list_groups() {
  cat <<EOF
Available groups:
  git-repo
  hf-cache
  nltk-data
  trippy-data
  trippy-r-code
  emoloop-models
EOF
}

check_variable() {
  local var_name="$1"
  local var_value="${!var_name}"
  if [[ -z "${var_value}" ]]; then
    echo "❌ Error: ${var_name} is not set."
    exit 1
  else
    echo "✅ ${var_name}=${var_value}"
  fi
}

run_rsync_pair() {
  local src="$1"
  local dest="$2"
  local extra_flags="$3"   # e.g. "--delete"

  echo "===================================================="
  echo "🔄 Syncing: '${src}' -> '${dest}'"

  # Create remote dir when destination is remote file path ending with '/'.
  # For local dest paths, create directory as needed.
  if [[ "$dest" != *":"* ]]; then
    mkdir -p "$dest" 2>/dev/null || true
  else
    # Remote destination: try to create directory on remote host
    local remote_host_prefix="${dest%%:*}"
    local remote_path="${dest#*:}"
    # Only attempt mkdir -p on remote when path ends with '/' (directory semantics)
    if [[ "$remote_path" == */ ]]; then
      if [[ -n "${DRY_RUN_FLAG}" ]]; then
        echo "(dry-run) ssh ${remote_host_prefix} mkdir -p '${remote_path}'"
      else
        ssh "${remote_host_prefix}" "mkdir -p '${remote_path}'" || true
      fi
    fi
  fi

  rsync -avhz --progress ${DRY_RUN_FLAG} ${extra_flags} \
    "$src" \
    "$dest"
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "❌ Error: rsync failed for '${src}' -> '${dest}' (exit $rc)"
  fi
  return $rc
}

# ------------------------------
# Parse args
# ------------------------------

DRY_RUN_FLAG=""
SELECTED_GROUPS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --group)
      shift
      [[ $# -gt 0 ]] || usage
      SELECTED_GROUPS+=("$1")
      shift
      ;;
    --dry-run)
      DRY_RUN_FLAG="--dry-run"
      shift
      ;;
    --list)
      list_groups
      exit 0
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "❌ Unknown argument: $1"
      usage
      ;;
  esac
done

if [[ ${#SELECTED_GROUPS[@]} -eq 0 ]]; then
  echo "❌ No --group provided."
  usage
fi

# ------------------------------
# Load env & common checks
# ------------------------------

if [[ -z "${TOPO_LLM_REPOSITORY_BASE_PATH}" ]]; then
  echo "❌ Error: TOPO_LLM_REPOSITORY_BASE_PATH is not set."
  exit 1
fi

source "${TOPO_LLM_REPOSITORY_BASE_PATH}/.env"

# Required across most groups
check_variable "TOPO_LLM_REPOSITORY_BASE_PATH"
check_variable "REMOTE_HOST"
check_variable "ZIM_USERNAME"

REMOTE_SSH_PREFIX="${ZIM_USERNAME}@${REMOTE_HOST}"

# ZIM_TOPO_LLM_REPOSITORY_BASE_PATH is used for repo-relative destinations
check_variable "ZIM_TOPO_LLM_REPOSITORY_BASE_PATH"

REMOTE_REPO_BASE="${REMOTE_SSH_PREFIX}:${ZIM_TOPO_LLM_REPOSITORY_BASE_PATH}"

overall_exit=0

PAIRS=()  # elements formatted as "src|dest|flags"

add_pair() {
  local src="$1"; local dest="$2"; local flags="$3"
  PAIRS+=("${src}|${dest}|${flags}")
}

add_group_pairs() {
  local grp="$1"
  case "$grp" in
    git-repo)
      check_variable "RSYNC_GIT_REPOSITORY_EXCLUDES_FILE"
      echo "TOPO_LLM_REPOSITORY_BASE_PATH=${TOPO_LLM_REPOSITORY_BASE_PATH}"
      add_pair \
        "${TOPO_LLM_REPOSITORY_BASE_PATH}/" \
        "${REMOTE_SSH_PREFIX}:/gpfs/project/${ZIM_USERNAME}/git-source/Topo_LLM/" \
        "-e ssh --exclude-from=${RSYNC_GIT_REPOSITORY_EXCLUDES_FILE}"
      ;;
    hf-cache)
      check_variable "LOCAL_HUGGINGFACE_CACHE_PATH"
      add_pair \
        "${LOCAL_HUGGINGFACE_CACHE_PATH}" \
        "${REMOTE_SSH_PREFIX}:/gpfs/project/${ZIM_USERNAME}/models/" \
        "--delete"
      ;;
    nltk-data)
      add_pair \
        "$HOME/nltk_data/" \
        "${REMOTE_SSH_PREFIX}:/home/${ZIM_USERNAME}/nltk_data/" \
        "--delete"
      ;;
    trippy-data)
      add_pair \
        "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_checkpoints/post_processed_cached_features/" \
        "${REMOTE_REPO_BASE}/data/models/trippy_checkpoints/post_processed_cached_features/" \
        ""
      add_pair \
        "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/post_processed_cached_features/" \
        "${REMOTE_REPO_BASE}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/post_processed_cached_features/" \
        ""
      ;;
    trippy-r-code)
      add_pair \
        "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/DO.run_train_and_run_eval.sh" \
        "${REMOTE_REPO_BASE}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/DO.run_train_and_run_eval.sh" \
        ""
      add_pair \
        "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/submit_run_eval_multiple_seeds.sh" \
        "${REMOTE_REPO_BASE}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/submit_run_eval_multiple_seeds.sh" \
        ""
      add_pair \
        "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/start_tensorboard.sh" \
        "${REMOTE_REPO_BASE}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints/start_tensorboard.sh" \
        ""
      add_pair \
        "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints_code/" \
        "${REMOTE_REPO_BASE}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints_code/" \
        ""
      ;;
    emoloop-models)
      add_pair \
        "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/EmoLoop/" \
        "${REMOTE_REPO_BASE}/data/models/EmoLoop/" \
        "--delete"
      ;;
    *)
      echo "❌ Unknown group: $grp"
      list_groups
      exit 1
      ;;
  esac
}

# Build pairs from requested groups
for grp in "${SELECTED_GROUPS[@]}"; do
  echo ""
  echo "================ BUILDING GROUP: ${grp} ================"
  add_group_pairs "$grp"
done

echo ""
echo "Pairs to sync: ${#PAIRS[@]}"
for p in "${PAIRS[@]}"; do
  IFS='|' read -r s d f <<< "$p"
  echo " - $s -> $d ${f:+(flags: $f)}"
done

echo ""
echo "================ EXECUTING RSYNC PAIRS ================"
for p in "${PAIRS[@]}"; do
  IFS='|' read -r src dest flags <<< "$p"
  run_rsync_pair "$src" "$dest" "$flags" || overall_exit=$?
done
echo "======================================================"

if [[ $overall_exit -ne 0 ]]; then
  echo ""
  echo "@@@ Completed with errors (exit $overall_exit)."
else
  echo ""
  echo "✅ All selected groups synced successfully."
fi

exit $overall_exit
