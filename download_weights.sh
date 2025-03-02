#!/bin/bash
# 设置环境变量
export HF_ENDPOINT=https://hf-mirror.com
export HUGGINGFACE_HUB_TOKEN=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
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# 安装或更新 huggingface_hub 包
pip install -U huggingface_hub

# 设置模型保存的基础路径
BASE_DIR="/h3cstore_ns/ydchen/code/MedicalFinder/models_HF"
mkdir -p "$BASE_DIR"

# 定义模型列表
declare -A models=(
  ["BioBert"]="dmis-lab/biobert-v1.1"
  ["BiomedCLIP"]=microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
)

# 下载每个模型
echo "开始下载模型..."

for model_name in "${!models[@]}"; do
  echo "正在下载模型: $model_name (${models[$model_name]})"
  huggingface-cli download \
    --token $HUGGINGFACE_HUB_TOKEN \
    --resume-download "${models[$model_name]}" \
    --local-dir "$BASE_DIR/$model_name" \
    --local-dir-use-symlinks False
  
  echo "模型 $model_name 下载完成"
done

echo "所有模型下载任务已完成"