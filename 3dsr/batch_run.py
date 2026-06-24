#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python 版本的批量处理脚本，用于为多个 Meningioma-SEG-CLASS-xxx 数据集生成距离索引图。
此脚本会遍历指定目录下的所有 Meningioma-SEG-CLASS-xxx 文件夹，
并为每个文件夹内的 tri_trainlist.txt 和 tri_testlist.txt 调用 multiprocess_create_dis_index.py。
"""

import os
import argparse
import subprocess
from pathlib import Path


def run_command(cmd, env=None):
    """
    运行系统命令并实时打印输出。
    Args:
        cmd (list): 命令及其参数的列表。
        env (dict, optional): 环境变量字典。
    """
    print(f"Executing: {' '.join(cmd)}")
    try:
        # 使用 subprocess.run 并实时输出
        result = subprocess.run(cmd, env=env, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        print(e.output)


def main():
    parser = argparse.ArgumentParser(description='Batch process distance index for multiple datasets.')
    parser.add_argument('--base_path', type=str, required=True,
                        help='Base path containing all Meningioma-SEG-CLASS-xxx folders.')
    parser.add_argument('--script_path', type=str, required=True,
                        help='Full path to the multiprocess_create_dis_index.py script.')
    parser.add_argument('--num_gpus', type=int, default=4,
                        help='Number of GPUs to use.')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers per GPU.')
    parser.add_argument('--downsample_ratio', type=float, default=2.0,
                        help='Downsampling ratio for distance map.')
    parser.add_argument('--sample_length', type=int, default=3,
                        help='Number of frames in each sequence (e.g., 3 for triplet).')

    args = parser.parse_args()

    # 获取所有 Meningioma-SEG-CLASS-xxx 文件夹
    base_path = Path(args.base_path)
    dataset_dirs = sorted(base_path.glob("la_*.nii.gz"))

    if not dataset_dirs:
        print(f"Error: No 'la_*.nii.gz' folders found in {args.base_path}")
        return

    print(f"Found {len(dataset_dirs)} datasets to process.")

    for dataset_dir in dataset_dirs:
        if not dataset_dir.is_dir():
            continue

        print(f"\n{'=' * 60}")
        print(f"Processing dataset: {dataset_dir.name}")
        print(f"{'=' * 60}")

        vimeo_triplet_dir = dataset_dir / "vimeo_triplet"
        if not vimeo_triplet_dir.exists():
            print(f"Warning: 'vimeo_triplet' folder not found in {dataset_dir}. Skipping...")
            continue

        # 处理训练集
        train_list = vimeo_triplet_dir / "tri_trainlist.txt"
        if train_list.exists():
            print(f">>> Processing {train_list} ...")
            cmd = [
                'python', args.script_path,
                '--num_gpus', str(args.num_gpus),
                '--num_workers', str(args.num_workers),
                '--path', str(dataset_dir),
                '--sample_list_path', 'vimeo_triplet/tri_trainlist.txt',
                '--downsample_ratio', str(args.downsample_ratio),
                '--sample_length', str(args.sample_length)
            ]
            # 设置环境变量
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, range(args.num_gpus)))
            run_command(cmd, env=env)
        else:
            print(f"Warning: '{train_list}' not found. Skipping train set.")

        # 处理测试集
        test_list = vimeo_triplet_dir / "tri_testlist.txt"
        if test_list.exists():
            print(f">>> Processing {test_list} ...")
            cmd = [
                'python', args.script_path,
                '--num_gpus', str(args.num_gpus),
                '--num_workers', str(args.num_workers),
                '--path', str(dataset_dir),
                '--sample_list_path', 'vimeo_triplet/tri_testlist.txt',
                '--downsample_ratio', str(args.downsample_ratio),
                '--sample_length', str(args.sample_length)
            ]
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, range(args.num_gpus)))
            run_command(cmd, env=env)
        else:
            print(f"Warning: '{test_list}' not found. Skipping test set.")

        print(f">>> Finished processing: {dataset_dir.name}\n")

    print("All datasets have been processed successfully!")


if __name__ == "__main__":
    main()