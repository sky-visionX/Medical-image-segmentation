#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert MSD Task05_Prostate (4D NIfTI: H x W x D x 2) into Vimeo90K-style triplet
folder structure for multiprocess_create_dis_index.py.

- channel 0 = T2
- channel 1 = ADC

Output per case:
<output_dir>/<case_name>/
  volume.nii.gz  (hardlink/copy from input)
  sequences/
    case_00001/
      im1.png im2.png im3.png
    case_00002/
      im1.png im2.png im3.png
    ...
  vimeo_triplet/
    tri_trainlist.txt   (absolute paths to case_xxxxx dirs)
    tri_testlist.txt
"""

import os
import argparse
import nibabel as nib
import numpy as np
import cv2
import shutil
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def normalize01(vol: np.ndarray) -> np.ndarray:
    """Normalize to [0, 1] using global min/max."""
    vmin = float(np.min(vol))
    vmax = float(np.max(vol))
    if vmax <= vmin + 1e-6:
        return np.zeros_like(vol, dtype=np.float32)
    return (vol - vmin) / (vmax - vmin)


def load_nifti_select_channel(img_path: str, channel: int) -> np.ndarray:
    """
    Load a NIfTI. If 4D, select the given modality channel.
    Returns a 3D volume (float32).
    Supports:
      - (H, W, D)               -> 그대로 반환
      - (H, W, D, C)            -> 选最后一维 wzh0901channel
      - (C, H, W, D)            -> 选第一维 channel
    """
    data = nib.load(img_path).get_fdata().astype(np.float32)

    if data.ndim == 3:
        return data

    if data.ndim == 4:
        # Common MSD style: (H, W, D, C)
        if data.shape[-1] >= 2:
            if channel < 0 or channel >= data.shape[-1]:
                raise ValueError(f"channel={channel} out of range for shape {data.shape}")
            return data[..., channel].astype(np.float32)

        # Alternative: (C, H, W, D)
        if data.shape[0] >= 2:
            if channel < 0 or channel >= data.shape[0]:
                raise ValueError(f"channel={channel} out of range for shape {data.shape}")
            return data[channel, ...].astype(np.float32)

    raise ValueError(f"Unsupported NIfTI shape: {data.shape} (ndim={data.ndim})")


def save_case_triplets(
    img_path: str,
    axis: int = 2,
    group_size: int = 3,
    overwrite: bool = False,
    train_ratio: float = 0.9,
    channel: int = 0,
) -> None:
    """
    Slice a single volume and save as Vimeo triplet format under the case folder.
    """
    case_root = os.path.dirname(img_path)
    case_name = os.path.basename(case_root)
    seq_dir = os.path.join(case_root, "sequences")
    vimeo_dir = os.path.join(case_root, "vimeo_triplet")

    if overwrite:
        shutil.rmtree(seq_dir, ignore_errors=True)
        shutil.rmtree(vimeo_dir, ignore_errors=True)

    os.makedirs(seq_dir, exist_ok=True)
    os.makedirs(vimeo_dir, exist_ok=True)

    # Load and select channel -> 3D
    data3d = load_nifti_select_channel(img_path, channel=channel)

    # Normalize to [0, 1]
    data3d = normalize01(data3d)

    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0/1/2, got {axis}")

    if data3d.ndim != 3:
        raise ValueError(f"Expected 3D after channel selection, got shape {data3d.shape}")

    depth = data3d.shape[axis]
    if depth < group_size:
        print(f"[WARN] {case_name}: depth={depth} < group_size={group_size}, skipped.")
        return

    num_groups = depth // group_size
    case_paths = []

    for g in range(num_groups):
        case_dir = os.path.join(seq_dir, f"case_{g+1:05d}")
        os.makedirs(case_dir, exist_ok=True)

        for j in range(group_size):
            idx = g * group_size + j
            sl = np.take(data3d, idx, axis=axis)  # should be 2D

            # Ensure 2D slice
            sl = np.squeeze(sl)
            if sl.ndim != 2:
                raise ValueError(f"{case_name}: slice is not 2D. slice shape={sl.shape}, axis={axis}, idx={idx}")

            sl8 = (sl * 255.0).astype(np.uint8)
            sl8 = np.ascontiguousarray(sl8)  # OpenCV safer

            out_png = os.path.join(case_dir, f"im{j+1}.png")
            ok = cv2.imwrite(out_png, sl8)
            if not ok:
                raise RuntimeError(f"cv2.imwrite failed: {out_png}")

        case_paths.append(os.path.abspath(case_dir))

    if not case_paths:
        print(f"[WARN] {case_name}: no valid groups generated.")
        return

    # Train/Test split (reproducible per-case)
    rng = random.Random(42)
    rng.shuffle(case_paths)
    split_idx = int(len(case_paths) * train_ratio)
    train_cases = case_paths[:split_idx]
    test_cases = case_paths[split_idx:]

    with open(os.path.join(vimeo_dir, "tri_trainlist.txt"), "w") as f:
        f.write("\n".join(train_cases))
    with open(os.path.join(vimeo_dir, "tri_testlist.txt"), "w") as f:
        f.write("\n".join(test_cases))

    ch_name = "T2" if channel == 0 else f"CH{channel}"
    if channel == 1:
        ch_name = "ADC"
    print(f"[INFO] {case_name} ({ch_name}): {len(train_cases)} train, {len(test_cases)} test | groups={len(case_paths)}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert MSD Task05_Prostate 4D NIfTI into Vimeo90K-style triplet datasetlabelprostate structure (single channel)."
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing prostate_*.nii.gz files (e.g., Task05_Prostate/imagesTr).")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for case folders. If not given, use input_dir.")
    parser.add_argument("--axis", type=int, default=2,
                        help="Slicing axis: 0=sagittal, 1=coronal, 2=axial (default: 2)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel threads (default: 4)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recreate output directories if they exist")
    parser.add_argument("--use_copy", action="store_true",
                        help="Use shutil.copy instead of hard link (required for cross-device)")
    parser.add_argument("--train_ratio", type=float, default=0.9,
                        help="Train/test split ratio (default: 0.9)")
    parser.add_argument("--channel", type=int, default=0, choices=[0, 1],
                        help="Select modality channel: 0=T2, 1=ADC (Task05_Prostate)")
    parser.add_argument("--pattern", type=str, default="prostate_*.nii.gz",
                        help="Glob pattern to find NIfTI files in input_dir (default: prostate_*.nii.gz)")
    parser.add_argument("--case_nii_name", type=str, default="volume.nii.gz",
                        help="Name of the NIfTI file to place inside each case folder (default: volume.nii.gz)")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    nii_files = sorted(input_dir.glob(args.pattern))
    nii_files = [f for f in nii_files if f.is_file()]

    if not nii_files:
        print(f"[ERROR] No NIfTI files matched '{args.pattern}' in {input_dir}")
        return

    print(f"[INFO] Found {len(nii_files)} NIfTI files.")
    print(f"[INFO] Using channel={args.channel} ({'T2' if args.channel==0 else 'ADC'}), axis={args.axis}")

    case_dirs = []
    for nii_path in nii_files:
        case_name = nii_path.name[:-7]  # remove '.nii.gz'
        case_dir = output_dir / case_name
        target_nii = case_dir / args.case_nii_name

        if args.overwrite and case_dir.exists():
            shutil.rmtree(case_dir)

        case_dir.mkdir(parents=True, exist_ok=True)

        # Link or copy NIfTI into case folder
        if not target_nii.exists():
            if args.use_copy:
                shutil.copy(nii_path, target_nii)
            else:
                try:
                    os.link(nii_path, target_nii)
                except (OSError, NotImplementedError):
                    print(f"[WARN] Hard link failed for {nii_path}, using copy.")
                    shutil.copy(nii_path, target_nii)

        case_dirs.append(str(case_dir))

    print(f"[INFO] Prepared {len(case_dirs)} case directories in {output_dir}")

    # Parallel processing
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for case_dir in case_dirs:
            img_path = os.path.join(case_dir, args.case_nii_name)
            futures.append(
                executor.submit(
                    save_case_triplets,
                    img_path,
                    args.axis,
                    3,
                    args.overwrite,
                    args.train_ratio,
                    args.channel
                )
            )

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            fut.result()

    print("[INFO] All cases processed successfully!")


if __name__ == "__main__":
    main()
