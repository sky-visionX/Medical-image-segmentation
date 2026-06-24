import os
import argparse
import nibabel as nib
import numpy as np
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import random

def normalize01(vol):
    """归一化到0~1"""
    vmin = float(np.min(vol))
    vmax = float(np.max(vol))
    if vmax <= vmin + 1e-6:
        return np.zeros_like(vol, dtype=np.float32)
    return (vol - vmin) / (vmax - vmin)

def save_case_triplets(img_path, out_root, axis=2, group_size=3, overwrite=False, train_ratio=0.9):
    """
    单个病例处理：
      - 读取 Meningioma_converted_preprocessed.nii.gz
      - 每 group_size 张为一组 → case_xxxxx 文件夹
      - 输出到: out_root/<case_name>/sequences 和 out_root/<case_name>/vimeo_triplet
    """
    case_in_root = os.path.dirname(img_path)           # 输入病例目录
    case_name = os.path.basename(case_in_root)

    # ===== 新增：输出病例根目录 =====
    case_out_root = os.path.join(out_root, case_name)
    seq_dir = os.path.join(case_out_root, "sequences")
    vimeo_dir = os.path.join(case_out_root, "vimeo_triplet")

    if overwrite:
        shutil.rmtree(case_out_root, ignore_errors=True)

    os.makedirs(seq_dir, exist_ok=True)
    os.makedirs(vimeo_dir, exist_ok=True)

    # 读取体数据
    data = nib.load(img_path).get_fdata().astype(np.float32)
    data = normalize01(data)
    depth = data.shape[axis]

    num_groups = depth // group_size
    case_paths = []

    for g in range(num_groups):
        case_dir = os.path.join(seq_dir, f"case_{g+1:05d}")
        os.makedirs(case_dir, exist_ok=True)

        for j in range(group_size):
            idx = g * group_size + j
            sl = np.take(data, indices=idx, axis=axis)
            sl8 = (sl * 255.0).astype(np.uint8)
            cv2.imwrite(os.path.join(case_dir, f"im{j+1}.png"), sl8)

        case_paths.append(os.path.abspath(case_dir))  # 绝对路径（指向 out_root 下）

    # train/test 划分
    random.shuffle(case_paths)
    split_idx = int(len(case_paths) * train_ratio)
    train_cases = case_paths[:split_idx]
    test_cases = case_paths[split_idx:]

    with open(os.path.join(vimeo_dir, "tri_trainlist.txt"), "w") as f:
        f.write("\n".join(train_cases))
    with open(os.path.join(vimeo_dir, "tri_testlist.txt"), "w") as f:
        f.write("\n".join(test_cases))

    print(f"[INFO] {case_name}: {len(train_cases)} train, {len(test_cases)} test -> {case_out_root}")

def main():
    parser = argparse.ArgumentParser(
        description="Batch: per-case processing of Meningioma_converted_preprocessed.nii.gz → case_xx PNG groups."
    )
    parser.add_argument("--root_dir", type=str, required=True,
                        help="包含 Meningioma-SEG-CLASS-xxx 子文件夹的目录（输入）")
    # ===== 新增：输出目录参数 =====
    parser.add_argument("--out_dir", type=str, required=True,
                        help="输出根目录（会在其中生成 out_dir/病例名/sequences 与 vimeo_triplet）")

    parser.add_argument("--axis", type=int, default=2,
                        help="切片轴 (0: sagittal, 1: coronal, 2: axial)，默认2")
    parser.add_argument("--workers", type=int, default=4,
                        help="并行线程数")
    parser.add_argument("--overwrite", action="store_true",
                        help="若存在则删除该病例在 out_dir 下的输出后重建")
    parser.add_argument("--train_ratio", type=float, default=0.9,
                        help="train 比例，默认0.9")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 找病例
    case_dirs = []
    for name in sorted(os.listdir(args.root_dir)):
        d = os.path.join(args.root_dir, name)
        if os.path.isdir(d) and name.startswith("Meningioma-SEG-CLASS-"):
            nii_path = os.path.join(d, "img.nii.gz")
            if os.path.exists(nii_path):
                case_dirs.append(d)

    print(f"[INFO] Found {len(case_dirs)} cases.")

    # 并行逐病例处理
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = []
        for d in case_dirs:
            img_path = os.path.join(d, "img.nii.gz")
            futures.append(
                ex.submit(save_case_triplets, img_path, args.out_dir, args.axis, 3, args.overwrite, args.train_ratio)
            )

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing cases"):
            fut.result()

    print(f"[INFO] All cases processed. Outputs in: {args.out_dir}")

if __name__ == "__main__":
    main()
