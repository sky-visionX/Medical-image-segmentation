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
def parse_args():
    parser = argparse.ArgumentParser(description="Convert NIfTI MRI dataset to VFI-compatible format.")
    parser.add_argument("--main_directory", type=str,
                        default=r"/home/dell/wzh/new/Meningioma_converted_preprocessed")
    parser.add_argument("--output_directory", type=str,
                        default=r"/home/dell/wzh/data/brain/1")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Proportion of images for validation/test list.")
    parser.add_argument("--axis", type=int, default=2,
                        help="Axis along which to slice the volume.")
    parser.add_argument("--overwrite", action='store_true',
                        help="Overwrite existing processed files.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of threads for parallel processing.")
    return parser.parse_args()

# 归一化函数
def normalize_image(image):
    return (image - image.min()) / (image.max() - image.min() + 1e-6)

# 切片函数：返回所有切片的完整路径列表
def slice_3d_to_images(data_3d, output_dir, axis=2):
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for i in range(data_3d.shape[axis]):
        sl = data_3d.take(i, axis=axis).astype(np.float32)
        sl = normalize_image(sl)
        sl = (sl * 255).astype(np.uint8)
        filename = f"slice_{i+1:03d}.jpg"
        path = os.path.join(output_dir, filename)
        cv2.imwrite(path, sl)
        paths.append(path)
    logging.info(f"Saved {len(paths)} slices to {output_dir}")
    return paths

# 单个病例处理函数：返回该病例的所有切片路径
def process_case_slices(folder, args):
    try:
        img_file = os.path.join(folder, "img.nii.gz")
        if not os.path.exists(img_file):
            logging.error(f"Missing label.nii.gz in {folder}")
            return []

        temp_dir = os.path.join(args.output_directory, "_temp_slices", os.path.basename(folder))
        os.makedirs(temp_dir, exist_ok=True)

        img_data = nib.load(img_file).get_fdata()
        return slice_3d_to_images(img_data, temp_dir, axis=args.axis)
    except Exception as e:
        logging.error(f"Failed processing {folder}: {e}")
        return []

if __name__ == "__main__":
    args = parse_args()
    # 日志配置
    os.makedirs(args.output_directory, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(args.output_directory, 'processing.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 获取病例文件夹列表
    cases = [os.path.join(args.main_directory, d)
             for d in os.listdir(args.main_directory)
             if os.path.isdir(os.path.join(args.main_directory, d)) and d.startswith("Meningioma-SEG-CLASS-")]
    logging.info(f"Found {len(cases)} cases.")

    # 并行切片
    all_slices = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_case_slices, folder, args) for folder in cases]
        for future in tqdm(futures, total=len(futures), desc="Processing cases"):
            paths = future.result()
            all_slices.extend(paths)

    logging.info(f"Total slices: {len(all_slices)}")

    # 创建输出图像目录
    volumes_dir = os.path.join(args.output_directory, "volumes")
    if args.overwrite and os.path.exists(volumes_dir):
        shutil.rmtree(volumes_dir)
    os.makedirs(volumes_dir, exist_ok=True)

    # 拷贝并重命名所有切片
    train_list = []
    test_list = []
    for idx, src in enumerate(sorted(all_slices)):
        dst_name = f"volume-{idx+1:06d}.jpg"
        dst_path = os.path.join(volumes_dir, dst_name)
        shutil.copy(src, dst_path)
        rel_path = os.path.join("volumes", dst_name)
        if random.random() < args.val_ratio:
            test_list.append(rel_path)
        else:
            train_list.append(rel_path)

    # 写入列表文件
    train_file = os.path.join(args.output_directory, "train_list.txt")
    test_file = os.path.join(args.output_directory, "test_list.txt")
    with open(train_file, 'w') as f:
        f.write("\n".join(train_list))
    with open(test_file, 'w') as f:
        f.write("\n".join(test_list))

    logging.info(f"Saved {len(train_list)} train and {len(test_list)} test entries.")
    print("[INFO] Conversion complete.")