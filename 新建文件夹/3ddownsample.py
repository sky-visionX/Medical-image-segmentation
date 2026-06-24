import os
import glob
import numpy as np
import nibabel as nib
from nilearn.image import resample_img


def downsample_nifti(input_folder, output_folder, target_shape=(40, 40)):
    # 如果输出文件夹不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取所有 .nii.gz 文件
    file_list = glob.glob(os.path.join(input_folder, "*.nii.gz"))

    for file_path in file_list:
        file_name = os.path.basename(file_path)
        print(f"正在处理: {file_name}")

        # 1. 加载原始图像
        img = nib.load(file_path)
        original_shape = img.shape

        # 2. 计算目标 shape (保持第3维深度不变)
        # 假设原始是 (256, 256, Z)，目标是 (40, 40, Z)
        if len(original_shape) == 3:
            new_shape = (target_shape[0], target_shape[1], original_shape[2])
        else:
            # 如果是 2D 的 NIfTI (256, 256)
            new_shape = target_shape

        # 3. 计算缩放后的 Affine 矩阵
        # 计算每个维度的缩放比例
        scale_factor = np.array(original_shape[:2]) / np.array(target_shape)
        new_affine = img.affine.copy()
        # 更新仿射矩阵中的体素大小 (Voxel Size)
        new_affine[:2, :2] = img.affine[:2, :2] * scale_factor.reshape(-1, 1)

        # 4. 执行重采样
        # interpolation='continuous' 适用于灰度图(CT/MRI)，'nearest' 适用于标签(Mask)
        resampled_img = resample_img(
            img,
            target_shape=new_shape,
            target_affine=new_affine,
            interpolation='continuous'
        )

        # 5. 保存结果
        save_path = os.path.join(output_folder, file_name)
        nib.save(resampled_img, save_path)
        print(f"已保存至: {save_path}")


if __name__ == "__main__":
    # 配置路径
    input_dir = '/home/dell/wzh/new/InterpAny-Clearer-main/datasetupsample'  # 替换为你的输入路径
    output_dir = '/home/dell/wzh/data/Task04_Hippocampus/gai/imagesTr'  # 替换为你的输出路径

    downsample_nifti(input_dir, output_dir)