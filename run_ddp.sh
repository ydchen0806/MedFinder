#!/bin/bash
pip install wandb -i https://pypi.tuna.tsinghua.edu.cn/simple

# 设置环境变量
export PYTHONPATH=/h3cstore_ns/ydchen/code/MedicalFinder:$PYTHONPATH
export WANDB_API_KEY=4dd8899dcb163e86d45644b7c896bfa7ec6af32b
export WANDB_PROJECT="MedFinder_a40"
export WANDB_NAME="250302"  # 与debug命令保持一致

# 定义参数
NUM_GPUS=8  # 根据debug命令修改为8个GPU
MASTER_PORT=12345  # 主端口
DATA_ROOT="/h3cstore_ns/CT_data/CT_retrieval"
OUTPUT_DIR="/h3cstore_ns/ydchen/code/MedicalFinder/output_0303"
BACKBONE="resnet50"
BATCH_SIZE=4  # 根据debug命令修改为4
EPOCHS=20  # 根据debug命令修改为20
LR=1e-4
ALPHA=1.0  # 损失权重

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 使用torchrun启动多GPU训练
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    /h3cstore_ns/ydchen/code/MedicalFinder/main_ddp.py \
    --data_root $DATA_ROOT \
    --output_dir $OUTPUT_DIR \
    --backbone $BACKBONE \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --alpha $ALPHA \
    --use_amp \
    --use_wandb \
    --seed 42 \
    --wandb_project $WANDB_PROJECT \
    --wandb_name $WANDB_NAME