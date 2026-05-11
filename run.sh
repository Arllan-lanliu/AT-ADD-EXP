gpu=0
config=conf/base.yaml
model_path=""            # 留空则自动从 config 的 out_fold 读取
RESUME=0                 # 1 = 从 checkpoint 继续训练（自动加载 model_path/config.yaml）
RUN_TRAIN=1              # 1 = 训练
RUN_SCORE=1              # 1 = 生成 eval 集预测分数（inference）
RUN_DEV_ANALYZE=0        # 1 = dev 集分析

# W&B: offline = 默认（免外网/证书问题）；online = 同步 wandb.ai（SSL 失败时 main_train 会自动改 offline）
wandb_mode=offline
wandb_project="AT-ADD-Baseline"
wandb_entity=""          # 可选：用户名或 team，留空则用默认 entity
wandb_run_name=""        # 可选：留空则用 out_fold 目录名

# ---------------------------------------------------------------------------
# Derive model_path from the YAML's out_fold if not set explicitly above.
# ---------------------------------------------------------------------------
if [[ -z "${model_path}" && -n "${config}" ]]; then
    model_path=$(python -c "
import yaml
with open('${config}') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('out_fold', './ckpt_t2/default_run'))
" 2>/dev/null)
fi
model_path=${model_path:-"./ckpt_t2/default_run"}

echo "============================================================"
echo " Config     : ${config}"
echo " Model path : ${model_path}"
echo " GPU        : ${gpu}"
echo " W&B mode   : ${wandb_mode}"
echo "============================================================"

# Common W&B CLI args for training
WB_ARGS=(--wandb_mode "${wandb_mode}" --wandb_project "${wandb_project}")
[[ -n "${wandb_entity}"   ]] && WB_ARGS+=(--wandb_entity "${wandb_entity}")
[[ -n "${wandb_run_name}" ]] && WB_ARGS+=(--wandb_run_name "${wandb_run_name}")

# =============================================================================
# Stage 1 — Training
# =============================================================================
if [[ "${RESUME}" == "1" ]]; then
    echo ""
    echo ">>> [Stage 1] Resume training from ${model_path}"
    # --resume loads config.yaml + latest.pt from model_path; no --config needed.
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONWARNINGS="ignore" \
    python main_train.py \
        --resume "${model_path}" \
        --gpu    "${gpu}" \
        "${WB_ARGS[@]}"
elif [[ "${RUN_TRAIN}" == "1" ]]; then
    echo ""
    echo ">>> [Stage 1] Training"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONWARNINGS="ignore" \
    python main_train.py \
        --config "${config}" \
        --gpu    "${gpu}" \
        "${WB_ARGS[@]}"
fi

# =============================================================================
# Stage 2 — Score generation (eval set)
# =============================================================================
if [[ "${RUN_SCORE}" == "1" ]]; then
    echo ""
    echo ">>> [Stage 2] Score generation (eval set)"
    PYTHONWARNINGS="ignore" \
    python scripts/inference.py \
        --model_path "${model_path}" \
        --gpu        "${gpu}" \
        --batch_size 160 \
        --eval_task  atadd-track2 \
        --threshold  0.5
fi

# =============================================================================
# Stage 3 — Dev-set analysis (optional)
# =============================================================================
if [[ "${RUN_DEV_ANALYZE}" == "1" ]]; then
    echo ""
    echo ">>> [Stage 3] Dev-set analysis"
    PYTHONWARNINGS="ignore" \
    python scripts/analyze.py \
        --model_path "${model_path}" \
        --gpu        "${gpu}" \
        --batch_size 32 \
        --eval_task  atadd-track2 \
        --metrics_only   # remove this flag to also generate attention heatmaps + t-SNE
fi
