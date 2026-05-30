#!/usr/bin/env bash
# Track1 multi-crop experiment. Mirrors ../1_t1_run.sh, but routes train/dev/eval
# through trunk_vote/*_multicrop.py so each crop is used in train and logits are
# averaged back to the original audio for dev/eval metrics.

set -e

gpu=4
PYTHON_BIN="${PYTHON_BIN:-python}"

CONFIGS=(
  trunk_vote/conf/xlsr_3_11_24_multicrop.yaml
)

config_single="trunk_vote/conf/xlsr_3_11_24_multicrop.yaml"

RESUME=0
RUN_TRAIN=1
RUN_DEV_ANALYZE=1
RUN_SCORE=1

wandb_mode=offline
wandb_project="3090-Track1"
wandb_entity=""
wandb_run_name=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${ROOT}" || exit 1

WB_ARGS=(--wandb_mode "${wandb_mode}" --wandb_project "${wandb_project}")
[[ -n "${wandb_entity}" ]] && WB_ARGS+=(--wandb_entity "${wandb_entity}")
[[ -n "${wandb_run_name}" ]] && WB_ARGS+=(--wandb_run_name "${wandb_run_name}")

_expand_out_fold() {
  "${PYTHON_BIN}" -c "
import yaml, sys
with open(sys.argv[1], encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('out_fold', './trunk_vote/ckpt_t1_layer/default_run'))
" "$1"
}

_run_one_exp() {
  local config="$1"
  local idx="$2"
  local total="$3"

  if [[ ! -f "${config}" ]]; then
    echo "[SKIP] 配置文件不存在: ${config}" >&2
    return 1
  fi

  local model_path
  model_path="$(_expand_out_fold "${config}")"
  model_path="${model_path:-./trunk_vote/ckpt_t1_layer/default_run}"

  echo ""
  echo "######################################################################"
  echo " Experiment ${idx} / ${total}"
  echo " Config       : ${config}"
  echo " Model path   : ${model_path}"
  echo " GPU          : ${gpu}"
  echo " Multi-crop   : train=crops, dev/eval=mean logits per audio"
  echo "######################################################################"

  if [[ "${RESUME}" == "1" ]]; then
    echo ""
    echo ">>> [Stage 1] Resume training from ${model_path}"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONWARNINGS="ignore" \
    "${PYTHON_BIN}" "${SCRIPT_DIR}/main_train_multicrop.py" \
        --resume "${model_path}" \
        --gpu    "${gpu}" \
        "${WB_ARGS[@]}"
  elif [[ "${RUN_TRAIN}" == "1" ]]; then
    echo ""
    echo ">>> [Stage 1] Training"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONWARNINGS="ignore" \
    "${PYTHON_BIN}" "${SCRIPT_DIR}/main_train_multicrop.py" \
        --config "${config}" \
        --gpu    "${gpu}" \
        "${WB_ARGS[@]}"
  fi

  if [[ "${RUN_DEV_ANALYZE}" == "1" ]]; then
    echo ""
    echo ">>> [Stage 2] Dev-set analysis"
    PYTHONWARNINGS="ignore" \
    "${PYTHON_BIN}" "${SCRIPT_DIR}/analyze_multicrop.py" \
        --model_path "${model_path}" \
        --gpu        "${gpu}" \
        --batch_size 160 \
        --eval_task  atadd-track1 \
        --metrics_only
  fi

  if [[ "${RUN_SCORE}" == "1" ]]; then
    echo ""
    echo ">>> [Stage 3] Score generation (eval set)"
    PYTHONWARNINGS="ignore" \
    "${PYTHON_BIN}" "${SCRIPT_DIR}/inference_multicrop.py" \
        --model_path "${model_path}" \
        --gpu        "${gpu}" \
        --batch_size 160 \
        --eval_task  atadd-track1 \
        --threshold  0.5
  fi

  echo ""
  echo ">>> Finished: ${config}"
}

declare -a RUN_LIST=()
if [[ ${#CONFIGS[@]} -gt 0 ]]; then
  for c in "${CONFIGS[@]}"; do
    [[ -z "${c}" ]] && continue
    [[ "${c}" =~ ^[[:space:]]*# ]] && continue
    RUN_LIST+=( "${c}" )
  done
fi
if [[ ${#RUN_LIST[@]} -eq 0 ]]; then
  if [[ -n "${config_single}" ]]; then
    RUN_LIST=( "${config_single}" )
  else
    echo "ERROR: CONFIGS 为空且 config_single 未设置。" >&2
    exit 1
  fi
fi

total=${#RUN_LIST[@]}
n=0
for config in "${RUN_LIST[@]}"; do
  n=$((n + 1))
  _run_one_exp "${config}" "${n}" "${total}"
done

echo ""
echo "Done. ${total} experiment(s) processed."
