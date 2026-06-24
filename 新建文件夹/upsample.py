import cv2
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm


def upscale_multiple_datasets(root_input, root_output, target_size=(256, 256)):
    """
    遍历根目录下所有子文件夹（如 hippocampus_001, 002...）
    并在新目录下完整重建 sequences/case_XXXXX 结构
    """
    input_base = Path(root_input)
    output_base = Path(root_output)

    # 递归查找所有图片文件 (im1.png, im2.png, im3.png 等)
    # 无论它在哪个子文件夹的序列里
    extensions = ("*.png", "*.jpg", "*.jpeg", "*.npy")
    all_files = []
    for ext in extensions:
        all_files.extend(list(input_base.rglob(ext)))

    print(f"在 {root_input} 及其子目录下共找到 {len(all_files)} 个文件。")

    for file_path in tqdm(all_files, desc="总进度"):
        # 1. 获取相对根目录的路径，例如: hippocampus_001/sequences/case_00001/im1.png
        relative_path = file_path.relative_to(input_base)

        # 2. 构造输出路径
        save_path = output_base / relative_path
        if save_path.suffix == '.npy':
            save_path = save_path.with_suffix('.png')

        # 3. 递归创建输出所需的各级文件夹
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # 4. 读取图像
        if file_path.suffix == '.npy':
            img = np.load(file_path)
        else:
            img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)

        if img is None:
            continue

        # 5. 医学影像预处理：归一化与通道对齐 (解决 RAFT 维度报错的关键)
        # 确保输入是符合 RAFT 预期的 uint8 RGB 格式 [cite: 189]
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # 如果是单通道(医学影像常见)，转换为 3 通道 [cite: 189]
        if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # 6. 上采样至 256x256
        # 解决由于输入太小导致 RAFT 内部特征图在池化时变为 0 的错误 [cite: 11, 151]
        img_res = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

        # 7. 写入新目录
        cv2.imwrite(str(save_path), img_res)


if __name__ == "__main__":
    # --- 路径配置 ---
    # 假设你的目录结构是：
    # /data/my_datasets/
    #    ├── hippocampus_001/
    #    ├── hippocampus_002/
    #    └── ...

    # 指向包含所有子数据集文件夹的那个根目录
    MY_ROOT_DIR = "/InterpAny-Clearer-main/datasetupsample"
    # 指向你想要生成的放大版数据集根目录
    MY_OUTPUT_DIR = "/InterpAny-Clearer-main/datasetupsample"

    upscale_multiple_datasets(MY_ROOT_DIR, MY_OUTPUT_DIR)
    print(f"\n全部处理完成！新数据集已保存在: {MY_OUTPUT_DIR}")