#!/usr/bin/env bash
# 删除 ckpt_t2 下每个实验目录中的 checkpoint/ 子文件夹
# 其他内容（args.json, atadd_model.pt, logs/, result/, wandb/ 等）保持不变

CKPT_ROOT="$(dirname "$0")/ckpt_t2"

if [ ! -d "$CKPT_ROOT" ]; then
    echo "目录不存在: $CKPT_ROOT"
    exit 1
fi

echo "扫描目录: $CKPT_ROOT"
echo "---"

count=0
for exp_dir in "$CKPT_ROOT"/*/; do
    ckpt_dir="${exp_dir}checkpoint"
    if [ -d "$ckpt_dir" ]; then
        echo "删除: $ckpt_dir"
        rm -rf "$ckpt_dir"
        count=$((count + 1))
    fi
done

echo "---"
echo "完成，共删除 ${count} 个 checkpoint/ 文件夹。"
