#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 la_xxx.nii.gz 格式的 3D 医学图像转换为 Vimeo90K 兼容的 triplet 结构，
以便后续使用 multiprocess_create_dis_index.py 生成距离索引图。
"""

import os
import argparse
import nibabel as nib
import numpy as np
import cv2
import shutil
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def normalize01(vol):
    """归一化到 [0, 1]"""
    vmin = float(np.min(vol))
    vmax = float(np.max(vol))
    if vmax <= vmin + 1e-6:
        return np.zeros_like(vol, dtype=np.float32)
    return (vol - vmin) / (vmax - vmin)


def save_case_triplets(img_path, axis=2, group_size=3, overwrite=False, train_ratio=0.9):
    """
    将单个体积数据切片并保存为 Vimeo triplet 格式。
    """
    case_root = os.path.dirname(img_path)
    case_name = os.path.basename(case_root)
    seq_dir = os.path.join(case_root, "sequences")
    vimeo_dir = os.path.join(case_root, "vimeo_triplet")

    if overwrite:
        shutil.rmtree(seq_dir, ignore_errors=True)
        shutil.rmtree(vimeo_dir, ignore_errors=True)
    os.makedirs(seq_dir, exist_ok=True)
    os.makedirs(vimeo_dir, exist_ok=True)

    # 加载并归一化
    data = nib.load(img_path).get_fdata().astype(np.float32)
    data = normalize01(data)
    depth = data.shape[axis]

    if depth < group_size:
        print(f"[WARN] {case_name}: depth={depth} < group_size={group_size}, skipped.")
        return

    num_groups = depth // group_size
    case_paths = []

    for g in range(num_groups):
        case_dir = os.path.join(seq_dir, f"case_{g+1:05d}")
        os.makedirs(case_dir, exist_ok=True)
        for j in range(group_size):
            idx = g * group_size + j
            sl = np.take(data, idx, axis=axis)
            sl8 = (sl * 255.0).astype(np.uint8)
            # 保存为灰度 PNG（单通道）
            cv2.imwrite(os.path.join(case_dir, f"im{j+1}.png"), sl8)
        case_paths.append(os.path.abspath(case_dir))

    if not case_paths:
        print(f"[WARN] {case_name}: no valid groups generated.")
        return

    # 随机划分训练/测试
    random.seed(42)  # 固定 seed 保证可复现
    random.shuffle(case_paths)
    split_idx = int(len(case_paths) * train_ratio)
    train_cases = case_paths[:split_idx]
    test_cases = case_paths[split_idx:]

    with open(os.path.join(vimeo_dir, "tri_trainlist.txt"), "w") as f:
        f.write("\n".join(train_cases))
    with open(os.path.join(vimeo_dir, "tri_testlist.txt"), "w") as f:
        f.write("\n".join(test_cases))

    print(f"[INFO] {case_name}: {len(train_cases)} train, {len(test_cases)} test")


def main():
    parser = argparse.ArgumentParser(
        description="Convert la_xxx.nii.gz files into Vimeo90K-style triplet datasetraw structure."
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing la_xxx.nii.gz files.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for case folders. If not given, use input_dir.")
    parser.add_argument("--axis", type=int, default=2,
                        help="Slicing axis: 0=sagittal, 1=coronal, 2=axial (default: 2)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel threads (default: 4)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recreate output directories if they exist")
    parser.add_argument("--use_copy", action="store_true",
                        help="Use shutil.copy instead of hard link (required for cross-device)")
    parser.add_argument("--train_ratio", type=float, default=0.9,
                        help="Train/test split ratio (default: 0.9)")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 找到所有 la_xxx.nii.gz 文件
    nii_files = sorted([
        f for f in input_dir.iterdir()
        if f.is_file() and f.name.startswith("Meningioma-SEG-CLASS-") and f.name.endswith(".nii.gz")
    ])

    if not nii_files:
        print(f"[ERROR] No 'la_xxx.nii.gz' files found in {input_dir}")
        return

    print(f"[INFO] Found {len(nii_files)} .nii.gz files.")

    # 为每个 .nii.gz 创建输出目录并准备数据
    case_dirs = []
    for nii_path in nii_files:
        case_name = nii_path.name[:-7]  # remove '.nii.gz'
        case_dir = output_dir / case_name
        target_nii = case_dir / "Meningioma_converted_preprocessed.nii.gz"

        if args.overwrite:
            if case_dir.exists():
                shutil.rmtree(case_dir)
        case_dir.mkdir(exist_ok=True)

        if not target_nii.exists():
            if args.use_copy:
                shutil.copy(nii_path, target_nii)
            else:
                try:
                    os.link(nii_path, target_nii)
                except (OSError, NotImplementedError):
                    print(f"[WARN] Hard link failed for {nii_path}, using copy.")
                    shutil.copy(nii_path, target_nii)

        case_dirs.append(str(case_dir))

    print(f"[INFO] Prepared {len(case_dirs)} case directories in {output_dir}")

    # 并行处理
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for case_dir in case_dirs:
            img_path = os.path.join(case_dir, "Meningioma_converted_preprocessed.nii.gz")
            futures.append(
                executor.submit(
                    save_case_triplets,
                    img_path,
                    args.axis,
                    3,
                    args.overwrite,
                    args.train_ratio
                )
            )

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            fut.result()

    print("[INFO] All cases processed successfully!")


if __name__ == "__main__":
    main()
