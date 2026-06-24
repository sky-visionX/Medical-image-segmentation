import nibabel as nib
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import argparse


def check_resolution(file_path):
    """打印 NIfTI 文件的分辨率信息"""
    img = nib.load(file_path)
    hdr = img.header
    print(f"File: {file_path}")
    print("Shape (x, y, z):", hdr.get_data_shape())
    zooms = hdr.get_zooms()
    print("Zooms (voxel size in mm) [x, y, z]:", zooms)
    print("Data type:", hdr.get_data_dtype())
    print("-" * 50)
    return img


def calculate_metrics(gt_data, pred_data):
    """计算 PSNR 和 SSIM（假设 shape 已对齐）"""
    gt_data = (gt_data - gt_data.min()) / (gt_data.max() - gt_data.min())
    pred_data = (pred_data - pred_data.min()) / (pred_data.max() - pred_data.min())

    avg_psnr = psnr(gt_data, pred_data, data_range=1.0)
    avg_ssim = ssim(gt_data, pred_data, channel_axis=None, data_range=1.0)

    print("\n[Metrics]")
    print(f"Avg PSNR: {avg_psnr:.2f} dB")
    print(f"Avg SSIM: {avg_ssim:.4f}")
    return {
        'psnr': avg_psnr,
        'ssim': avg_ssim
    }


def identify_hr_lr(path1, path2, calc_metrics=False):
    """识别哪个是 HR 哪个是 LR，并可选地计算 PSNR/SSIM"""
    img1 = nib.load(path1)
    img2 = nib.load(path2)

    zoom1 = img1.header.get_zooms()
    zoom2 = img2.header.get_zooms()

    data1 = img1.get_fdata()
    data2 = img2.get_fdata()

    # 取最小 depth，防止 shape 不一致导致报错
    min_depth = min(data1.shape[2], data2.shape[2])
    data1_cropped = data1[:, :, :min_depth]
    data2_cropped = data2[:, :, :min_depth]

    # 比较 Z 轴分辨率（slice thickness）
    if zoom1[2] < zoom2[2]:
        hr_file = path1
        lr_file = path2
        hr_data = data1_cropped
        lr_data = data2_cropped
    else:
        hr_file = path2
        lr_file = path1
        hr_data = data2_cropped
        lr_data = data1_cropped

    print("[INFO] Resolution Comparison:")
    print(f"Z-axis resolution of '{hr_file}' is smaller → High-resolution (HR)")
    print(f"Z-axis resolution of '{lr_file}' is larger → Low-resolution (LR)")

    metrics = None
    if calc_metrics:
        print(f"[INFO] Calculating PSNR/SSIM between '{hr_file}' and '{lr_file}'")
        try:
            metrics = calculate_metrics(hr_data, lr_data)
        except Exception as e:
            print(f"[ERROR] Could not calculate metrics: {e}")

    return hr_file, lr_file, metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Identify which NIfTI file is high-resolution or low-resolution.")
    parser.add_argument("--file1", type=str, required=True, help="Path to first .nii/.nii.gz file (LR or HR)")
    parser.add_argument("--file2", type=str, required=True, help="Path to second .nii/.nii.gz file (LR or HR)")
    parser.add_argument("--calc_metrics", action="store_true", help="Calculate PSNR/SSIM between two files")
    args = parser.parse_args()

    print("=" * 60)
    print("MRI Resolution Identification Tool")
    print("=" * 60)

    # Step 1: 检查两个文件的基本信息
    check_resolution(args.file1)
    check_resolution(args.file2)

    # Step 2: 判断哪张是 HR 哪张是 LR
    hr_path, lr_path, metrics = identify_hr_lr(args.file1, args.file2, calc_metrics=args.calc_metrics)

    print("/n[Final Result]")
    print(f"High-resolution file: {hr_path}")
    print(f"Low-resolution file: {lr_path}")

    # Step 3: 如果启用指标计算
    if args.calc_metrics and metrics:
        print(f"PSNR: {metrics['psnr']:.2f} dB")
        print(f"SSIM: {metrics['ssim']:.4f}")