#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import os.path as osp
import subprocess
import sys
from typing import List


def is_dataset_root(d: str, train_list_rel: str) -> bool:
    """
    判断 d 是否为一个数据集根目录：
    - 必须存在 sequences/
    - 必须存在 train_list_rel（默认 vimeo_triplet/tri_trainlist.txt）
    """
    return osp.isdir(osp.join(d, "sequences")) and osp.exists(osp.join(d, train_list_rel))


def find_dataset_roots(base: str, train_list_rel: str, recursive: bool) -> List[str]:
    """
    支持两种：
    1) base 本身就是 datasetraw root（例如 datasetdemo）
    2) base 是父目录，下面有多个 datasetraw root
       - recursive=False：只扫描一层子目录
       - recursive=True：递归扫描
    """
    base = osp.abspath(base)
    roots: List[str] = []

    # 情况 1：base 本身就是 datasetraw root
    if is_dataset_root(base, train_list_rel):
        return [base]

    # 情况 2：base 是父目录
    if not osp.isdir(base):
        return []

    if not recursive:
        # 只扫描一层
        for name in os.listdir(base):
            d = osp.join(base, name)
            if osp.isdir(d) and is_dataset_root(d, train_list_rel):
                roots.append(d)
        return sorted(roots)

    # 递归扫描：找到 root 后不再深入（避免把 sequences/case_xx 当 root）
    for cur, dirs, files in os.walk(base):
        if is_dataset_root(cur, train_list_rel):
            roots.append(cur)
            dirs[:] = []  # stop descending
    return sorted(roots)


def run_multiprocess(
    dataset_root: str,
    mp_script: str,
    num_gpus: int,
    num_workers: int,
    downsample_ratio: float,
    sample_length: int,
    list_relpath: str,
) -> int:
    """
    调用 multiprocess_create_dis_index.py
    注意：list_relpath 必须是相对 dataset_root 的路径（例如 vimeo_triplet/tri_trainlist.txt）
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(num_gpus)))

    cmd = [
        sys.executable, mp_script,
        "--num_gpus", str(num_gpus),
        "--num_workers", str(num_workers),
        "--path", dataset_root,
        "--sample_list_path", list_relpath,
        "--downsample_ratio", str(downsample_ratio),
        "--sample_length", str(sample_length),
    ]

    print("\n[RUN]", " ".join(cmd))
    p = subprocess.Popen(cmd, env=env)
    code = p.wait()
    if code != 0:
        print(f"[ERROR] exit_code={code}  datasetraw={dataset_root}  list={list_relpath}")
    return code


def main():
    """
    用法示例：

    1) base_path 本身就是 datasetdemo/
    python batch_compute_dt_final.py \
      --base_path /path/to/datasetdemo \
      --mp_script /path/to/multiprocess_create_dis_index.py \
      --num_gpus 2 --num_workers 1 --downsample_ratio 2.0 --sample_length 3

    2) base_path 是父目录，下面有很多 dataset_xxx/
    python batch_compute_dt_final.py \
      --base_path /path/to/all_datasets \
      --mp_script /path/to/multiprocess_create_dis_index.py \
      --num_gpus 2 --num_workers 1 --downsample_ratio 2.0 --sample_length 3

    3) 父目录更深，用递归：
    python batch_compute_dt_final.py ... --recursive
    """
    parser = argparse.ArgumentParser(description="Batch compute Dt (dis_index) for datasets with sequences/ + vimeo_triplet/*.txt")
    parser.add_argument("--base_path", type=str, required=True,
                        help="数据集根目录 或 包含多个数据集根目录的父目录")
    parser.add_argument("--mp_script", type=str, required=True,
                        help="multiprocess_create_dis_index.py 的完整路径")

    parser.add_argument("--num_gpus", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--downsample_ratio", type=float, default=2.0)
    parser.add_argument("--sample_length", type=int, default=3)

    # 你的 list 文件相对 dataset_root 的路径（按你截图结构）
    parser.add_argument("--train_list", type=str, default="vimeo_triplet/tri_trainlist.txt")
    parser.add_argument("--test_list", type=str, default="vimeo_triplet/tri_testlist.txt")

    parser.add_argument("--recursive", action="store_true",
                        help="递归扫描 base_path 下所有子目录来寻找数据集根目录")

    args = parser.parse_args()

    train_rel = args.train_list
    test_rel = args.test_list

    roots = find_dataset_roots(args.base_path, train_rel, args.recursive)
    if not roots:
        print(f"[ERROR] 没找到任何数据集根目录。要求：必须包含 sequences/ 且存在 {train_rel}")
        raise SystemExit(1)

    print(f"[INFO] Found {len(roots)} datasetraw root(s).")

    for ds in roots:
        print("\n" + "=" * 80)
        print("[DATASET ROOT]", ds)
        print("=" * 80)

        # train
        train_abs = osp.join(ds, train_rel)
        if osp.exists(train_abs):
            run_multiprocess(
                dataset_root=ds,
                mp_script=args.mp_script,
                num_gpus=args.num_gpus,
                num_workers=args.num_workers,
                downsample_ratio=args.downsample_ratio,
                sample_length=args.sample_length,
                list_relpath=train_rel,
            )
        else:
            print("[SKIP] train list not found:", train_abs)

        # test（不存在就跳过）
        test_abs = osp.join(ds, test_rel)
        if osp.exists(test_abs):
            run_multiprocess(
                dataset_root=ds,
                mp_script=args.mp_script,
                num_gpus=args.num_gpus,
                num_workers=args.num_workers,
                downsample_ratio=args.downsample_ratio,
                sample_length=args.sample_length,
                list_relpath=test_rel,
            )
        else:
            print("[SKIP] test list not found:", test_abs)

    print("\n[INFO] All done.")


if __name__ == "__main__":
    main()