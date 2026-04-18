#!/usr/bin/env bash
# =============================================================================
# AT-ADD Track-2  —  Training & Inference launcher
#
# Usage:
#   Edit the variables in each section, then run:
#       bash run.sh
#
# Model registry (--model):
#   Conventional:
#     aasist            Raw-waveform AASIST (no SSL)
#     specresnet        Mel-spectrogram + ResNet-18
#
#   Frozen SSL + AASIST (❄️  SSL encoder fixed):
#     fr-w2v2aasist     XLS-R-300M  + AASIST
#     fr-wavlmaasist    WavLM-Large + AASIST
#     fr-mertaasist     MERT-330M   + AASIST
#
#   Fine-tuned single-SSL + AASIST (🔥 end-to-end):
#     ft-w2v2aasist     XLS-R-300M  + AASIST
#     ft-wavlmaasist    WavLM-Large + AASIST
#     ft-mertaasist     MERT-330M   + AASIST
#     ft-beatsaasist    BEATs       + AASIST
#     ft-clapaasist    CLAP        + AASIST
#     ft-xlsr_sls       XLS-R + layer-weighted SLS head
#
#   Fine-tuned dual-SSL + AASIST (🔥🔥 two encoders):
#     ft-xlsrwavlmaasist    XLS-R + WavLM   + fusion + AASIST
#     ft-xlsrmertaasist     XLS-R + MERT    + fusion + AASIST
#     ft-xlsrbeatsaasist   XLS-R + BEATs   + fusion + AASIST
#     ft-xlsrclapaasist     XLS-R + CLAP    + fusion + AASIST
#
#   Prompt-tuned SSL + AASIST (🎯 frozen backbone, learnable prompts):
#     pt-w2v2aasist     PT-XLS-R    + AASIST
#     pt-wavlmaasist    PT-WavLM    + AASIST
#     pt-mertaasist     PT-MERT     + AASIST
#     wpt-w2v2aasist    WPT-XLS-R   + AASIST  (wavelet prompts)
#     wpt-wavlmaasist   WPT-WavLM   + AASIST
#     wpt-mertaasist    WPT-MERT    + AASIST
#
# Fusion methods for dual-SSL models (--fusion):
#   cat_linear    Concat + Linear (default, most parameters)
#   gated         Soft-gate interpolation
#   cross_attn    Bidirectional cross-attention
#   film          Feature-wise Linear Modulation (y modulates x)
#   type_aware    Dynamic per-type weights + auxiliary type-clf loss
#   proj512_cat   Each stream → 512-d, then concat (no joint linear)
#   add           Element-wise addition (zero extra parameters)
#
# Audio augmentation (train set only; dev is never augmented):
#   --aug_speech  P  : RawBoost(algo=5) on speech samples  (0=off)
#   --aug_sound   P  : RawBoost(algo=5) on sound  samples  (0=off)
#   --aug_singing P  : RawBoost(algo=5) on singing samples (0=off)
#   --aug_music   P  : augmentation    on music   samples  (0=off)
#   --music_aug_method  pitch_shift | spec_augment
#   --train_class_rawboost : apply RawBoost only to sound & singing classes
# =============================================================================

# ── GPU & experiment identity ─────────────────────────────────────────────────
gpu=3

# ── Model selection ───────────────────────────────────────────────────────────
# Change this to switch the frontend/backend combination.
# Examples:
#   model=ft-xlsrmertaasist   (dual-SSL, most common)
#   model=ft-w2v2aasist       (single fine-tuned SSL)
#   model=pt-w2v2aasist       (prompt-tuned, fewer trainable params)
model=ft-xlsrmertaasist

# ── Fusion method (dual-SSL models only) ──────────────────────────────────────
# Ignored for single-SSL models and ft-xlsrbeats_aasist.
# Options: cat_linear | gated | cross_attn | film | type_aware | proj512_cat | add
fusion=cat_linear

# filter types subset of audio types for train AND dev: speech,sound,music,singing. 
# Omit or empty = use all types.
filter_types=sound

# ── Output folder ─────────────────────────────────────────────────────────────
model_path=./ckpt_t2/${model}_${fusion}_${filter_types}

# ── Training hyperparameters ──────────────────────────────────────────────────
num_epochs=5
batch_size=16
lr=0.000001
interval=2        # LR decay every N epochs
seed=1234

# ── Evaluation schedule ───────────────────────────────────────────────────────
eval_steps=500          # mid-epoch eval every N steps (0 = epoch-end only)
eval_warmup_steps=2000  # skip eval for the first N global steps
patience=5              # early-stop after N evals without improvement, 0 = no early-stop

# ── Per-type audio augmentation probabilities ─────────────────────────────────
# Set to 0.0 to disable. Values in [0.0, 1.0].
aug_speech=0.0
aug_sound=0.0
aug_singing=0.0
aug_music=0.0
# Method for music augmentation: pitch_shift | spec_augment
music_aug_method=spec_augment

# ── Weights & Biases ─────────────────────────────────────────────────────────
wandb_project=AT-ADD-Track2

# =============================================================================
# Training
# =============================================================================
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONWARNINGS="ignore" \
python main_train.py \
    --gpu            ${gpu} \
    --train_task     atadd-track2 \
    --model          ${model} \
    --fusion         ${fusion} \
    --out_fold       ${model_path} \
    --num_epochs     ${num_epochs} \
    --batch_size     ${batch_size} \
    --lr             ${lr} \
    --interval       ${interval} \
    --seed           ${seed} \
    --num_workers    4 \
    --eval_steps          ${eval_steps} \
    --eval_warmup_steps   ${eval_warmup_steps} \
    --patience            ${patience} \
    --filter_types   ${filter_types} \
    --aug_speech     ${aug_speech} \
    --aug_sound      ${aug_sound} \
    --aug_singing    ${aug_singing} \
    --aug_music      ${aug_music} \
    --music_aug_method ${music_aug_method} \
    --wandb_project  ${wandb_project} \
    --wandb_run_name "${model_path}"

# =============================================================================
# Inference  —  generate submission scores
# =============================================================================
PYTHONWARNINGS="ignore" \
python generate_score.py \
    --gpu        ${gpu} \
    --batch_size 160 \
    --eval_task  atadd-track2 \
    --model_path ${model_path} \
    --threshold  0.5

# =============================================================================
# Resume from checkpoint (uncomment to use)
# =============================================================================
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# PYTHONWARNINGS="ignore" \
# python main_train.py \
#     --continue_training \
#     --gpu        ${gpu} \
#     --train_task atadd-track2 \
#     --model      ${model} \
#     --fusion     ${fusion} \
#     --out_fold   ${model_path} \
#     --num_epochs ${num_epochs} \
#     --batch_size ${batch_size} \
#     --lr         ${lr} \
#     --interval   ${interval} \
#     --seed       ${seed}

# =============================================================================
# Sync W&B offline run (uncomment after training)
# =============================================================================
# wandb sync ${model_path}/wandb/

# =============================================================================
# Example: run multiple experiments in parallel (one per GPU)
# =============================================================================
# gpu=0 model=ft-xlsrmertaasist fusion=cat_linear  model_path=./ckpt_t2/${model}_${fusion}  bash run.sh &
# gpu=1 model=ft-xlsrmertaasist fusion=proj512_cat model_path=./ckpt_t2/${model}_${fusion}  bash run.sh &
# gpu=2 model=ft-xlsrmertaasist fusion=add         model_path=./ckpt_t2/${model}_${fusion}  bash run.sh &
# gpu=3 model=ft-xlsrclapaasist fusion=cat_linear  model_path=./ckpt_t2/${model}_${fusion}  bash run.sh &
# wait
