#!/bin/bash
pip install wandb -i https://pypi.tuna.tsinghua.edu.cn/simple


# 定义参数
NUM_GPUS=8  # 使用8个GPU进行评估
MASTER_PORT=12346  # 使用不同的端口以避免冲突
DATA_ROOT="/h3cstore_ns/CT_data/CT_retrieval"
OUTPUT_DIR="/h3cstore_ns/ydchen/code/MedicalFinder/evaluation_results_0302"
BACKBONE="resnet50"
BATCH_SIZE=4  # 降低批处理大小以减少内存使用
MODEL_PATH="/h3cstore_ns/ydchen/code/MedicalFinder/output_0302/best_model.pt"  # 指定要评估的模型路径

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 使用torchrun启动多GPU评估
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    /h3cstore_ns/ydchen/code/MedicalFinder/main_ddp.py \
    --data_root $DATA_ROOT \
    --output_dir $OUTPUT_DIR \
    --backbone $BACKBONE \
    --batch_size $BATCH_SIZE \
    --eval_only \
    # --model_path $MODEL_PATH \
    --use_amp \
    # --use_wandb \
    --seed 42
