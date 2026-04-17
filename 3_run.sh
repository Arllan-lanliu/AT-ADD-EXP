gpu=2
model_name=ft-xlsrclapaasist
model_path=./ckpt_t2/${model_name}_cat_linear


# XLSR-300M [B, T, 1024] + CLAP/HTSAT [B, T, 1024] (freq-avg + interp) → cat → Linear(2048→1024) → AASIST
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONWARNINGS="ignore" python main_train.py --gpu ${gpu} --train_task atadd-track2 \
        --num_epochs 5 --num_workers 4 --batch_size 16 --lr 0.000001  --interval 2 --seed 1234 \
        --model ${model_name} --out_fold ${model_path} --wandb_project AT-ADD-Track2 --wandb_run_name "${model_path}" \
        --fusion cat_linear \
        --eval_steps 500 \
        --eval_warmup_steps 2000 \
        --patience 5

# wandb sync ${model_path}/wandb/

# 生成submission文件
PYTHONWARNINGS="ignore" python generate_score.py --gpu ${gpu} --batch_size 160 --eval_task atadd-track2 --model_path ${model_path} --threshold 0.5



#从上次中断的地方继续训练
# python main_train.py --continue_training --train_task atadd-track2 --model ${model_name} --num_epochs 1 --interval 2 --seed 1234 --batch_size 14 --lr 0.000001 --out_fold ${model_path}
