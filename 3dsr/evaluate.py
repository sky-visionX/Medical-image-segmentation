import os
import nibabel as nib
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import argparse


def load_nii_file(file_path):
    """加载 .nii 或 .nii.gz 文件"""
    img = nib.load(file_path)
    data = img.get_fdata()
    return data


def normalize_image(image):
    """将图像归一化到 [0, 1]"""
    return (image - image.min()) / (image.max() - image.min())


def calculate_metrics(gt_data, pred_data, verbose=False):
    """
    计算所有切片的 PSNR 和 SSIM，支持 3D volume
    """
    if gt_data.shape != pred_data.shape:
        raise ValueError(f"Shapes do not match: GT {gt_data.shape}, Pred {pred_data.shape}")

    start_slice = 10
    end_slice = -30
    gt_data = gt_data[..., start_slice:end_slice]
    pred_data = pred_data[..., start_slice:end_slice]
    gt_data = normalize_image(gt_data)
    pred_data = normalize_image(pred_data)

    psnr_values = []
    ssim_values = []

    depth = gt_data.shape[2]
    for i in range(depth):
        gt_slice = gt_data[:, :, i]
        pred_slice = pred_data[:, :, i]

        # 确保不是全黑或无效切片
        if np.max(gt_slice) == 0 and np.max(pred_slice) == 0:
            print(f"[INFO] Skipping slice {i} (both are black)")
            continue

        # 计算指标
        p = psnr(gt_slice, pred_slice, data_range=1.0)
        s = ssim(gt_slice, pred_slice, data_range=1.0)

        psnr_values.append(p)
        ssim_values.append(s)

        if verbose:
            print(f"Slice {i}: PSNR={p:.4f}, SSIM={s:.4f}")

    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)

    print(f"Average PSNR: {avg_psnr:.4f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")

    return {
        'psnr': avg_psnr,
        'ssim': avg_ssim
    }


def save_metrics(metrics, output_file="metrics.txt"):
    """将指标保存到文件"""
    with open(output_file, "w") as f:
        f.write(f"Avg PSNR: {metrics['psnr']:.4f} dB\n")
        f.write(f"Avg SSIM: {metrics['ssim']:.4f}\n")
    print(f"[INFO] Metrics saved to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate PSNR and SSIM between two NIfTI files.")
    parser.add_argument("--input", type=str, required=True, help="Path to input/interpolated MRI (.nii.gz)")
    parser.add_argument("--gt", type=str, required=True, help="Path to ground truth MRI (.nii.gz)")
    parser.add_argument("--verbose", action="store_true", help="Print metrics per slice")
    args = parser.parse_args()

    print(f"[INFO] Loading input file: {args.input}")
    pred_data = load_nii_file(args.input)

    print(f"[INFO] Loading GT file: {args.gt}")
    gt_data = load_nii_file(args.gt)

    print("[INFO] Calculating metrics...")
    metrics = calculate_metrics(gt_data, pred_data, verbose=args.verbose)

    # 保存结果
    output_dir = os.path.dirname(args.input) or "."
    save_metrics(metrics, os.path.join(output_dir, "psnr_ssim_results.txt"))