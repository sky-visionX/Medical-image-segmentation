import os
import nibabel as nib
import numpy as np
import cv2
import subprocess
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from lpips import LPIPS
import torch
import glob
from scipy.ndimage import zoom
# ----------------------------
# Step 1: 加载和切片 MRI 数据
# ----------------------------

def load_nii_file(file_path):
    img = nib.load(file_path)
    data = img.get_fdata()
    return data

def slice_3d_to_2d(data_3d, axis=2):
    slices = [data_3d.take(i, axis=axis) for i in range(data_3d.shape[axis])]
    return slices

def normalize_image(image):
    return (image - image.min()) / (image.max() - image.min())

def save_slices_to_images(slices, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for i, sl in enumerate(slices):
        path = os.path.join(output_dir, f"slice_{i:04d}.png")
        cv2.imwrite(path, (normalize_image(sl) * 255).astype(np.uint8))
        paths.append(path)
    return paths

# ----------------------------
# Step 2: 调用 VFI 模型插帧
# ----------------------------

def interpolate_slice_pair(img0, img1, model="RIFE", variant="DR",
                           checkpoint="checkpoints/RIFE/DR-RIFE-pro",
                           save_dir="interpolated_slices",num=[1],iters=0):
    num_str = ' '.join(map(str, args.num)) if isinstance(args.num, list) else str(args.num)
    iters_str = ' '.join(map(str, args.iters)) if isinstance(args.iters, list) else str(args.iters)
    cmd = f"""
    python inference_img.py \
      --img0 {img0} \
      --img1 {img1} \
      --model {model} \
      --variant {variant} \
      --checkpoint {checkpoint} \
      --save_dir {save_dir} \
      --num {num_str} \
      --iters {iters_str}
    """
    if iters > 0:
        cmd += f" --iters {iters}"

    print(f"[INFO] Running command: {cmd}")
    subprocess.run(cmd, shell=True, check=True)
    # 自动获取目录下所有 .png 文件
    results = glob.glob(os.path.join(save_dir, "*.png"))
    if not results:
        raise FileNotFoundError(f"No interpolation results found in {save_dir}")

    # 假设插值结果保存在 save_dir 中
    result_path = os.path.join(save_dir, "001.png")
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"Interpolation result not found: {result_path}")
    return result_path
    # 取中间帧作为最终插值结果（推荐）
    mid_index = len(results) // 2
    interpolated_path = sorted(results)[mid_index]
    return sorted(results)[0]
# ----------------------------
# Step 3: 堆叠插值图像为 3D volume
# ----------------------------

def stack_2d_slices_to_3d(slice_paths="/home/dell/wzh/new/InterpAny-Clearer-main/visual"):
    slices = []
    for path in slice_paths:
        print(f"[DEBUG] Loading image: {path}")
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        slices.append(img / 255.0)
    return np.stack(slices, axis=2)

def save_3d_volume(volume, affine, output_path):
    new_img = nib.Nifti1Image(volume, affine)
    nib.save(new_img, output_path)
    print(f"[INFO] Saved 3D volume to {output_path}")

# ----------------------------
# Step 4: PSNR / SSIM / LPIPS 评估
# ----------------------------

# loss_fn = LPIPS(net='alex')

def calculate_metrics(gt, pred):
    print("[DEBUG] Starting metrics calculation...")
    print(f"[DEBUG] GT shape: {gt.shape}")
    print(f"[DEBUG] Pred shape: {pred.shape}")

    if gt.shape != pred.shape:
        raise ValueError("Ground truth and prediction have different shapes.")

    # 归一化处理
    gt = (gt - gt.min()) / (gt.max() - gt.min())
    pred = (pred - pred.min()) / (pred.max() - pred.min())

    psnr_values = []
    ssim_values = []

    for i in range(gt.shape[2]):
        gtsl = gt[:, :, i]
        predsl = pred[:, :, i]

        # 调试每一层切片是否正常
        print(f"[DEBUG] Calculating metrics for slice {i}...")

        psnr_val = psnr(gtsl, predsl, data_range=1.0)
        ssim_val = ssim(gtsl, predsl, data_range=1.0)

        psnr_values.append(psnr_val)
        ssim_values.append(ssim_val)

        print(f"[DEBUG] Slice {i}: PSNR={psnr_val:.4f}, SSIM={ssim_val:.4f}")

    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)

    print(f"[INFO] Avg PSNR: {avg_psnr:.2f} dB")
    print(f"[INFO] Avg SSIM: {avg_ssim:.4f}")

    return {
        'psnr': avg_psnr,
        'ssim': avg_ssim
    }
    # gt = (gt - gt.min()) / (gt.max() - gt.min())
    # pred = (pred - pred.min()) / (pred.max() - pred.min())
    #
    # psnr_values = []
    # ssim_values = []
    #
    # for i in range(gt.shape[2]):
    #     gtsl = gt[:, :, i]
    #     predsl = pred[:, :, i]
    #     psnr_values.append(psnr(gtsl, predsl, data_range=1.0))
    #     ssim_values.append(ssim(gtsl, predsl, data_range=1.0))
    #
    # avg_psnr = np.mean(psnr_values)
    # avg_ssim = np.mean(ssim_values)
    #
    # # LPIPS
    # gt_tensor = torch.tensor(gt).unsqueeze(0).unsqueeze(0).float()
    # pred_tensor = torch.tensor(pred).unsqueeze(0).unsqueeze(0).float()
    # # with torch.no_grad():
    # #     avg_lpips = loss_fn(gt_tensor, pred_tensor).item()
    #
    # print(f"Avg PSNR: {avg_psnr:.2f} dB")
    # print(f"Avg SSIM: {avg_ssim:.4f}")
    # # print(f"Avg LPIPS: {avg_lpips:.4f}")
    #
    # return {
    #     'psnr': avg_psnr,
    #     'ssim': avg_ssim,
    #     # 'lpips': avg_lpips
    # }

# ----------------------------
# Step 5: 主函数
# ----------------------------

def process_mri_with_vfi(
    low_res_path,
    gt_path=None,
    model="RIFE",
    variant="DR",
    checkpoint="checkpoints/RIFE/DR-RIFE-pro",
    temp_dir="temp_slices",
    results_dir="interpolated_slices",
    output_path="results/interpolated_result.nii.gz",
    num=[1],
    iters = 0
):
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Step 1: Load and slice
    low_data = load_nii_file(low_res_path)
    low_slices = slice_3d_to_2d(low_data)
    slice_paths = save_slices_to_images(low_slices, temp_dir)

    # Step 2: Interpolate each pair
    interpolated_paths = []
    for i in range(len(slice_paths) - 1):
        img0 = slice_paths[i]
        img1 = slice_paths[i + 1]
        interp_dir = os.path.join(results_dir, f"pair_{i:04d}")
        interp_path = os.path.join(interp_dir, "001.png")

        if os.path.exists(interp_path):
            print(f"[INFO] Skipping existing interpolation at {interp_path}")
            interpolated_paths.append(interp_path)
            continue

        if gt_path and os.path.exists(gt_path):
            print(f"[INFO] Evaluating with GT: {gt_path}")
            gt_data = load_nii_file(gt_path)
            print(f"[DEBUG] GT shape: {gt_data.shape}")
            interpolated_3d = stack_2d_slices_to_3d(interpolated_paths)
            print(f"[DEBUG] Pred shape: {interpolated_3d.shape}")

        os.makedirs(interp_dir, exist_ok=True)
        interpolate_slice_pair(img0, img1, model=model, variant=variant,
                               checkpoint=checkpoint, save_dir=interp_dir, num=num)
        interpolated_paths.append(interp_path)

    # Step 3: 合并原始切片与插值结果（交替插入）
    combined_paths = []

    # 循环插入原始 + 插值帧
    for i in range(len(slice_paths) - 1):
        combined_paths.append(slice_paths[i])  # 原始帧 A
        combined_paths.append(interpolated_paths[i])  # 插值帧 A+1
    combined_paths.append(slice_paths[-1])  # 加入最后一张原始帧

    print(f"[DEBUG] Total slices after interpolation: {len(combined_paths)}")

    # 现在堆叠的是原始 + 插值帧
    # interpolated_3d = stack_2d_slices_to_3d(combined_paths)
    # # Step 3.5: Upsample Z-axis to match GT depth (e.g., 216)
    # target_depth = 216
    # print(f"[INFO] Upsampling interpolated volume from {interpolated_3d.shape[2]} to {target_depth} slices")
    # upsampled_3d = zoom(interpolated_3d, (1, 1, target_depth / interpolated_3d.shape[2]), order=1)
    # save_3d_volume(upsampled_3d, np.eye(4), output_path)

    # Step 4: Evaluate if ground truth is provided
    metrics = {}
    if gt_path and os.path.exists(gt_path):
        gt_data = load_nii_file(gt_path)
        metrics = calculate_metrics(gt_data, interpolated_3d)
        with open("results/metrics.txt", "w") as f:
            f.write(f"PSNR: {metrics['psnr']}\n")
            f.write(f"SSIM: {metrics['ssim']}\n")
            f.write(f"LPIPS: {metrics['lpips']}\n")

    return output_path, metrics


# ----------------------------
# Step 6: 命令行入口
# ----------------------------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="MRI Video Frame Interpolation Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Path to input low-res MRI .nii.gz file")
    parser.add_argument("--gt", type=str, default=None, help="Path to ground truth high-res MRI (optional)")
    parser.add_argument("--model", type=str, default="RIFE", choices=["RIFE", "IFRNet", "AMT-S", "EMA-VFI"], help="VFI model")
    parser.add_argument("--variant", type=str, default="DR", choices=["T", "D", "TR", "DR"], help="Model variant")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/RIFE/DR-RIFE-pro", help="Model checkpoint path")
    parser.add_argument("--save_dir", type=str, default="interpolated_slices", help="Directory to save interpolated images")
    parser.add_argument("--num", type=int, nargs='+', default=[1], help="Number of extracted images per pair")
    parser.add_argument("--output", type=str, default="/home/dell/wzh/new/InterpAny-Clearer-main/result/interpolated_result.nii.gz",help="Output interpolated MRI path")
    parser.add_argument("--iters", type=int, default=0, help="Number of recursive iterations for interpolation")
    args = parser.parse_args()

    output_nii_path = args.output  # 最终输出的 .nii.gz 文件
    intermediate_save_dir = args.save_dir  # 插值图像保存的目录

    process_mri_with_vfi(
        low_res_path=args.input,
        gt_path=args.gt,
        model=args.model,
        variant=args.variant,
        checkpoint=args.checkpoint,
        temp_dir="temp_slices",  # 切片临时目录
        results_dir=intermediate_save_dir,  # 使用用户指定的 --save_dir
        output_path=output_nii_path,  # 最终输出文件
        iters = args.iters
    )
