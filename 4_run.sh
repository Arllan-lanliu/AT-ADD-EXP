gpu=3
model_name=ft-xlsrmertaasist
epochs=3
fusion=film
model_path=./ckpt_t2/${model_name}_${fusion}_ep${epochs}

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONWARNINGS="ignore" python main_train.py --gpu ${gpu} --train_task atadd-track2 \
#         --num_epochs ${epochs} --num_workers 4 --batch_size 16 --lr 0.000001 --interval 2 --seed 1234 \
#         --model ${model_name} --out_fold ${model_path} --wandb_project AT-ADD-Track2 --wandb_run_name "${model_path}" \
#         --fusion ${fusion}
        
wandb sync ${model_path}/wandb/offline-run-20260417_004146-h4c6hju4

# 生成submission文件
# PYTHONWARNINGS="ignore" python generate_score.py --gpu ${gpu} --batch_size 160 --eval_task atadd-track2 --model_path ${model_path} --threshold 0.5




# dev
# PYTHONWARNINGS="ignore" python analyze_dev_attention.py --model_path ${model_path} --gpu ${gpu} --batch_size 20 --eval_task atadd-track2 --eval_audio /data/liulan/workspace/dataset/at_add_track2/dev --label_path /data/liulan/workspace/dataset/at_add_track2/labels/dev.csv --audio_len 64600 --attn_frames 200 --score_suffix "_dev" --out_dir ${model_path}/analysis_dev_attention --tsne_perplexity 30.0 --tsne_seed 1234

#从上次中断的地方继续训练
# python main_train.py --continue_training --train_task atadd-track2 --model ${model_name} --num_epochs 1 --interval 2 --seed 1234 --batch_size 14 --lr 0.000001 --out_fold ${model_path}
