import os
import nibabel as nib
import numpy as np
import cv2
import subprocess
import glob
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# ----------------------------
# Utility functions
# ----------------------------
def load_nii_file(file_path):
    """Load a NIfTI file and return its dataset2 array."""
    img = nib.load(file_path)
    return img.get_fdata()


def slice_3d_to_2d(data_3d, axis=2):
    """Slice a 3D volume into a list of 2D arrays along the given axis."""
    return [data_3d.take(i, axis=axis) for i in range(data_3d.shape[axis])]


def normalize_image(image):
    """Normalize a 2D image to [0,1]."""
    return (image - image.min()) / (image.max() - image.min())


def save_slices_to_images(slices, output_dir):
    """Save a list of 2D arrays as PNG files, return list of paths."""
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for i, sl in enumerate(slices):
        img = normalize_image(sl)
        path = os.path.join(output_dir, f"slice_{i:04d}.png")
        cv2.imwrite(path, (img * 255).astype(np.uint8))
        paths.append(path)
    return paths


def stack_2d_slices_to_3d(slice_paths):
    """Stack a list of PNG slice paths into a 3D NumPy array."""
    arrs = []
    for p in slice_paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        arrs.append(img.astype(np.float32) / 255.0)
    return np.stack(arrs, axis=2)


def save_3d_volume(volume, affine, output_path):
    """Save a 3D NumPy array as a NIfTI file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    new_img = nib.Nifti1Image(volume, affine)
    nib.save(new_img, output_path)
    print(f"[INFO] Saved 3D volume to {output_path}")


def interpolate_slice_pair(img0, img1, model, variant, checkpoint, save_dir, num, iters):
    """Call external inference script to generate intermediate frame."""
    num_str = ' '.join(map(str, num))
    cmd = (
        f"python inference_img.py --img0 {img0} --img1 {img1}"
        f" --model {model} --variant {variant} --checkpoint {checkpoint}"
        f" --save_dir {save_dir} --num {num_str}"
    )
    if iters > 0:
        cmd += f" --iters {iters}"

    print(f"[INFO] Running command: {cmd}")
    subprocess.run(cmd, shell=True, check=True)
    results = glob.glob(os.path.join(save_dir, "*.png"))
    if not results:
        raise FileNotFoundError(f"No interpolation results in {save_dir}")
    pred_path = os.path.join(save_dir, "001.png")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Interpolation result not found: {pred_path}")
    return pred_path

# ----------------------------
# Main processing
# ----------------------------
# ----------------------------
# Main processing
# ----------------------------
def process_mri_with_vfi(
    low_res_path,
    model="RIFE", variant="DR", checkpoint="checkpoints/RIFE/DR-RIFE-pro",
    temp_dir="temp_slices", results_dir="interpolated_slices",
    output_path="results/interpolated_result.nii.gz",
    num=[1], iters=0, max_slices=184  # max_slices 控制截取多少张切片
):
    # 1) Load and slice 3D volume
    data3d = load_nii_file(low_res_path)
    slices = slice_3d_to_2d(data3d, axis=2)

    # 截取前 max_slices 张切片（假设数据至少有 max_slices 张）
    slices = slices[:max_slices]

    # 保存截取后的切片
    slice_paths = save_slices_to_images(slices, temp_dir)

    # 2) Interpolate each missing slice and compute metrics
    psnr_list, ssim_list = [], []
    interpolated_paths = []

    for i in range(1, len(slice_paths) - 1):
        img0 = slice_paths[i-1]
        gt_path = slice_paths[i]
        img2 = slice_paths[i+1]
        pair_dir = os.path.join(results_dir, f"pair_{i-1:04d}")
        os.makedirs(pair_dir, exist_ok=True)
        pred_path = os.path.join(pair_dir, "001.png")

        # run interpolation if needed
        if not os.path.exists(pred_path):
            interpolate_slice_pair(
                img0, img2, model, variant, checkpoint, pair_dir, num, iters
            )
        interpolated_paths.append(pred_path)

        # load and normalize for metric
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        gt   = cv2.imread(gt_path,   cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

        p = psnr(gt, pred, data_range=1.0)
        s = ssim(gt, pred, data_range=1.0)
        print(f"[METRIC] slice {i}: PSNR={p:.4f} dB, SSIM={s:.4f}")
        psnr_list.append(p)
        ssim_list.append(s)

    # 3) Report average
    avg_p = np.mean(psnr_list)
    avg_s = np.mean(ssim_list)
    print(f"[RESULT] Average PSNR: {avg_p:.4f} dB, Average SSIM: {avg_s:.4f}")

    # 4) Stack original + interpolated and save 3D volume
    combined = []
    combined.append(slice_paths[0])
    for i in range(len(slice_paths) - 1):
        combined.append(interpolated_paths[i-1])
        combined.append(slice_paths[i])
    combined.append(slice_paths[-1])

    vol3d = stack_2d_slices_to_3d(combined)
    save_3d_volume(vol3d, np.eye(4), output_path)

    # Optionally save metrics to file
    metrics_file = os.path.join(os.path.dirname(output_path), "metrics.txt")
    with open(metrics_file, 'w') as f:
        for idx, (p, s) in enumerate(zip(psnr_list, ssim_list), start=1):
            f.write(f"slice {idx}: PSNR={p:.4f}, SSIM={s:.4f}\n")
        f.write(f"\nAverage PSNR={avg_p:.4f}, SSIM={avg_s:.4f}\n")
    print(f"[INFO] Metrics saved to {metrics_file}")

# ----------------------------
# Command-line interface
# ----------------------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="3D MRI Frame Interpolation with per-slice PSNR/SSIM metrics"
    )
    parser.add_argument("--input",  type=str, required=True, help="Low-res input .nii.gz path")
    parser.add_argument("--model",  type=str, default="RIFE",   help="VFI model (RIFE|IFRNet|AMT-S|EMA-VFI)")
    parser.add_argument("--variant",type=str, default="DR",     help="Variant (T|D|TR|DR)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/RIFE/DR-RIFE-pro",
                        help="Model checkpoint directory")
    parser.add_argument("--save_dir", type=str, default="interpolated_slices",
                        help="Directory to save interpolated PNGs")
    parser.add_argument("--output", type=str, default="results/interpolated_result.nii.gz",
                        help="Output 3D NIfTI path")
    parser.add_argument("--num", type=int, nargs='+', default=[1], help="Number of frames to generate per pair")
    parser.add_argument("--iters", type=int, default=0, help="Recursion iterations for interpolation")
    args = parser.parse_args()

    process_mri_with_vfi(
        low_res_path=args.input,
        model=args.model,
        variant=args.variant,
        checkpoint=args.checkpoint,
        temp_dir="temp_slices",
        results_dir=args.save_dir,
        output_path=args.output,
        num=args.num,
        iters=args.iters
    )
