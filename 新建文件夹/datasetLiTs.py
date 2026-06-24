import os
import shutil
import nibabel as nib
import numpy as np
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import argparse
import random
import logging

# 参数解析器
parser = argparse.ArgumentParser(description="Convert NIfTI MRI Meningioma_converted_preprocessedraw to VFI-compatible format.")
parser.add_argument("--main_directory", type=str, default=r"/home/dell/wzh/new/LiTs")
parser.add_argument("--output_directory", type=str, default=r"/home/dell/wzh/new/InterpAny-Clearer-main/dataset2")
parser.add_argument("--val_ratio", type=float, default=0.1)
parser.add_argument("--axis", type=int, default=2)
parser.add_argument("--overwrite", action='store_true', help="Overwrite existing processed cases")
parser.add_argument("--workers", type=int, default=4, help="Number of threads for parallel processing")
args = parser.parse_args()

# 创建输出目录结构
os.makedirs(args.output_directory, exist_ok=True)
seq_dir = os.path.join(args.output_directory, "sequences")
os.makedirs(seq_dir, exist_ok=True)

# 列表文件路径
train_list_file = os.path.join(args.output_directory, "tri_trainlist.txt")
test_list_file = os.path.join(args.output_directory, "tri_testlist.txt")

# 归一化函数
def normalize_image(image):
    return (image - image.min()) / (image.max() - image.min() + 1e-6)

# 切片函数：返回所有切片的完整路径列表
def slice_3d_to_images(data_3d, output_dir, axis=2, naming='Meningioma_converted_preprocessed'):
    os.makedirs(output_dir, exist_ok=True)
    slices = []
    for i in range(data_3d.shape[axis]):
        sl = data_3d.take(i, axis=axis).astype(np.float32)
        sl = normalize_image(sl)
        sl = (sl * 255).astype(np.uint8)
        path = os.path.join(output_dir, f"im{i+1}.png")
        cv2.imwrite(path, sl)
        slices.append(path)
    print(f"[INFO] Saved {len(slices)} slices to {output_dir}")
    return slices

# 单个 NIfTI 文件处理函数
def process_single_volume(nii_path):
    try:
        filename = os.path.basename(nii_path).replace('.nii.gz', '').replace('.nii', '')
        temp_slice_dir = os.path.join(args.output_directory, "_temp_slices", filename)
        os.makedirs(temp_slice_dir, exist_ok=True)

        img_data = nib.load(nii_path).get_fdata()
        slice_paths = slice_3d_to_images(img_data, temp_slice_dir, axis=args.axis, naming='im')
        return slice_paths

    except Exception as e:
        print(f"[ERROR] Failed processing {nii_path}: {str(e)}")
        logging.error(f"Failed processing {nii_path}: {str(e)}")
        return []

# 主程序入口
if __name__ == "__main__":
    logging.basicConfig(
        filename=os.path.join(args.output_directory, 'processing.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 获取所有 .nii 或 .nii.gz 文件
    nii_files = [
        os.path.join(args.main_directory, f)
        for f in os.listdir(args.main_directory)
        if f.startswith("volume-") and (f.endswith(".nii") or f.endswith(".nii.gz"))
    ]

    print(f"[INFO] Found {len(nii_files)} NIfTI volumes.")

    global_slice_paths = []

    # 多线程处理
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_single_volume, f) for f in nii_files]
        for future in tqdm(futures, total=len(futures)):
            slice_paths = future.result()
            if slice_paths:
                global_slice_paths.extend(slice_paths)

    print(f"[INFO] Total slices collected: {len(global_slice_paths)}")

    # 清空已有的 sequences 组目录（如果 overwrite 启用）
    if args.overwrite:
        for item in os.listdir(seq_dir):
            item_path = os.path.join(seq_dir, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)

    # 每 3 张图像组成一个 group
    group_size = 3
    groups = [global_slice_paths[i:i + group_size] for i in range(0, len(global_slice_paths), group_size)]

    train_group_paths = []
    test_group_paths = []

    for idx, group in enumerate(groups):
        if len(group) < group_size:
            print(f"[WARNING] Skipping incomplete group {idx + 1} (only {len(group)} images)")
            continue

        group_name = f"group_{idx + 1:05d}"
        group_dir = os.path.join(seq_dir, group_name)
        os.makedirs(group_dir, exist_ok=True)

        for i, src_path in enumerate(group):
            dst_path = os.path.join(group_dir, f"im{i + 1}.png")
            shutil.copy(src_path, dst_path)

        if random.random() < args.val_ratio:
            test_group_paths.append(os.path.join("sequences", group_name))
        else:
            train_group_paths.append(os.path.join("sequences", group_name))

    # 写入训练测试列表
    with open(train_list_file, 'w') as f:
        f.write('\n'.join(train_group_paths))

    with open(test_list_file, 'w') as f:
        f.write('\n'.join(test_group_paths))

    print(f"[INFO] Dataset saved to {args.output_directory}")
    print(f"[INFO] Train groups: {len(train_group_paths)}, Test groups: {len(test_group_paths)}")
