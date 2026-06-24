import os
import numpy as np
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import argparse

def load_image(file_path):
    """以灰度图方式读取，并转为 float32"""
    img = imread(file_path, as_gray=True)
    return img.astype(np.float32)

def normalize_image(img):
    return (img - img.min()) / (img.max() - img.min())

def calculate_2d_metrics(gt, pred, verbose=False):
    gt = normalize_image(gt)
    pred = normalize_image(pred)
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: {gt.shape} vs {pred.shape}")
    p = psnr(gt, pred, data_range=1.0)
    s = ssim(gt, pred, data_range=1.0)
    if verbose:
        print(f"PSNR: {p:.4f} dB")
        print(f"SSIM: {s:.4f}")
    return {'psnr': p, 'ssim': s}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compute PSNR/SSIM between two 2D images"
    )
    parser.add_argument("--img1", required=True, help="Path to first image (png/jpg/…)")
    parser.add_argument("--img2", required=True, help="Path to second image")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    gt = load_image(args.img1)
    pred = load_image(args.img2)
    metrics = calculate_2d_metrics(gt, pred, verbose=args.verbose)

    out = os.path.splitext(args.img1)[0] + "_metrics.txt"
    with open(out, "w") as f:
        f.write(f"PSNR: {metrics['psnr']:.4f} dB\n")
        f.write(f"SSIM: {metrics['ssim']:.4f}\n")
    print(f"[INFO] Results saved to {out}")
