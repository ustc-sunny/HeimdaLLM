#!/bin/bash
# 为所有新数据集生成 uniform_client_1000 分区

# 切换到项目根目录
cd "$(dirname "$0")/../.."

DATA_DIR=./fednlp_data

# 新数据集列表
datasets=(
    "20news"
    "cnn_dailymail"
    "ploner"
    "semeval_2010_task8"
    "sst_2"
)

echo "========================================="
echo "为以下数据集生成 uniform_client_1000 分区:"
echo "${datasets[@]}"
echo "========================================="
echo ""

# 对每个数据集执行分区生成
for dataset in "${datasets[@]}"
do
    echo "----------------------------------------"
    echo "处理数据集: $dataset"
    echo "----------------------------------------"
    
    python -m data.advanced_partition.generate_uniform_client_1000 \
        --data_file ${DATA_DIR}/data_files/${dataset}_data.h5 \
        --partition_file ${DATA_DIR}/partition_files/${dataset}_partition.h5 \
        --client_num 1000 \
        --seed 42
    
    if [ $? -eq 0 ]; then
        echo "✅ $dataset 分区生成成功"
    else
        echo "❌ $dataset 分区生成失败"
    fi
    echo ""
done

echo "========================================="
echo "所有数据集处理完成！"
echo "========================================="
