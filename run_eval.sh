
atadd_t2_train_audio=/data/liulan/workspace/dataset/at_add_track2/train
atadd_t2_train_label=/data/liulan/workspace/dataset/at_add_track2/labels/train.csv
atadd_t2_dev_audio=/data/liulan/workspace/dataset/at_add_track2/dev
atadd_t2_dev_label=/data/liulan/workspace/dataset/at_add_track2/labels/dev.csv
atadd_t2_eval_audio=/data/liulan/workspace/dataset/at_add_track2/eval
xlsr=/data/liulan/workspace/huggingface/wav2vec2-xls-r-300m
wavlm=/data/liulan/workspace/huggingface/wavlm-large
mert=/data/liulan/workspace/huggingface/MERT-v1-330M
beats=/data/liulan/workspace/huggingface/BEATs_iter3
clap=/data/liulan/workspace/huggingface/larger_clap_music_and_speech    
#----------------------------------------------------------------------------
#!/usr/bin/env bash
gpu=0
model_name=ft-beatsaasist
# # speech uses RawBoost algo=3 (SSI_additive_noise); sound uses default algo=5 (LnL+ISD)
# model_path=./ckpt_t2/${model_name}_aug_speech0.4_sound0.8

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONWARNINGS="ignore" python main_train.py --gpu ${gpu} --train_task atadd-track2 \
#         --atadd_t2_train_audio ${atadd_t2_train_audio} --atadd_t2_train_label ${atadd_t2_train_label} --atadd_t2_dev_audio ${atadd_t2_dev_audio} --atadd_t2_dev_label ${atadd_t2_dev_label} --atadd_t2_eval_audio ${atadd_t2_eval_audio} \
#         --xlsr ${xlsr} --wavlm ${wavlm} --mert ${mert} --beats ${beats} --clap ${clap} \
#         --num_epochs 10 --num_workers 4 --batch_size 24 --lr 0.000001 --interval 2 --seed 1234 \
#         --model ${model_name} --out_fold ${model_path} --wandb_project AT-ADD-Track2 --wandb_run_name "${model_path}" \
#         --aug_speech 0.4 \
#         --aug_sound 0.8 \
#         --eval_steps 0 \
#         --no_dev_subsample

# wandb sync ${model_path}/wandb/


model_path=/data/liulan/workspace/released_models/AT-ADD-Baseline/ckpt_t2/ft-beatsaasist_aug_speech0.2_sound0.8
# 在整个dev集上评估，输出各类型/整体 Macro-F1，不做 attention 热图和 t-SNE
PYTHONWARNINGS="ignore" python analyze_dev_attention.py \
        --gpu ${gpu} --batch_size 40 --eval_task atadd-track2 \
        --model_path ${model_path} \
        --eval_audio ${atadd_t2_dev_audio} \
        --label_path ${atadd_t2_dev_label} \
        --out_dir ${model_path}/analysis_dev \
        --metrics_only


# 生成submission文件
PYTHONWARNINGS="ignore" python generate_score.py --gpu ${gpu} --batch_size 40 --eval_task atadd-track2 --model_path ${model_path} --threshold 0.5



# 从上次中断的地方继续训练
# python main_train.py --continue_training --train_task atadd-track2 --model ${model_name} --num_epochs 10 --interval 2 --seed 1234 --batch_size 24 --lr 0.000001 --out_fold ${model_path}
