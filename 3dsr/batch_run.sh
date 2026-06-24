#!/bin/bash

# 设置您的基础路径，所有 Meningioma-SEG-CLASS-xxx 文件夹都位于此路径下
BASE_PATH="/home/dell/wzh/new/InterpAny-Clearer-main/Meningioma_converted_preprocessed_lr1"  # 请修改为您的实际路径，例如 /mnt/disks/ssd0 或 /home/dell/wzh/new/InterpAny-Clearer-main/

# 设置 GPU 和 Worker 数量
NUM_GPUS=2   # 根据您的机器配置修改
NUM_WORKERS=1 # 每个 GPU 的进程数

# 设置下采样比例和样本长度
DOWNSAMPLE_RATIO=2.0
SAMPLE_LENGTH=3 # 对于 triplet 数据集

# 遍历所有以 "Meningioma-SEG-CLASS-" 开头的文件夹
for dataset_dir in "$BASE_PATH"/Meningioma-SEG-CLASS-*; do
    # 检查是否为目录
    if [ -d "$dataset_dir" ]; then
        echo "============================================"
        echo "Processing dataset: $dataset_dir"
        echo "============================================"

        # 进入该数据集目录
        cd "$dataset_dir" || { echo "Failed to enter directory: $dataset_dir"; continue; }

        # 检查 vimeo_triplet 文件夹是否存在
        if [ ! -d "vimeo_triplet" ]; then
            echo "Warning: 'vimeo_triplet' folder not found in $dataset_dir. Skipping..."
            continue
        fi

        # 处理 train.txt
        if [ -f "vimeo_triplet/tri_trainlist.txt" ]; then
            echo ">>> Processing tri_trainlist.txt ..."
            CUDA_VISIBLE_DEVICES=0,1,2,3 python /path/to/your/multiprocess_create_dis_index.py \
                --num_gpus $NUM_GPUS \
                --num_workers $NUM_WORKERS \
                --path "$dataset_dir" \
                --sample_list_path "vimeo_triplet/tri_trainlist.txt" \
                --downsample_ratio $DOWNSAMPLE_RATIO \
                --sample_length $SAMPLE_LENGTH
        else
            echo "Warning: 'tri_trainlist.txt' not found. Skipping train set."
        fi

        # 处理 test.txt
        if [ -f "vimeo_triplet/tri_testlist.txt" ]; then
            echo ">>> Processing tri_testlist.txt ..."
            CUDA_VISIBLE_DEVICES=0,1,2,3 python /path/to/your/multiprocess_create_dis_index.py \
                --num_gpus $NUM_GPUS \
                --num_workers $NUM_WORKERS \
                --path "$dataset_dir" \
                --sample_list_path "vimeo_triplet/tri_testlist.txt" \
                --downsample_ratio $DOWNSAMPLE_RATIO \
                --sample_length $SAMPLE_LENGTH
        else
            echo "Warning: 'tri_testlist.txt' not found. Skipping test set."
        fi

        echo ">>> Finished processing: $dataset_dir"
        echo ""
    fi
done

echo "All datasets have been processed!"