#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scheme A (FIXED):
Label 不做光流、不做 warp
只按 image 的 slice 数复制最近 label
并且：完全复制 image 的几何信息（spacing / direction）
"""

import argparse
from pathlib import Path
import numpy as np
import nibabel as nib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_in", required=True)
    parser.add_argument("--image_ref", required=True)
    parser.add_argument("--label_out", required=True)
    parser.add_argument("--pattern", default="prostate_*.nii.gz")
    parser.add_argument("--axis", type=int, default=2)
    args = parser.parse_args()

    label_in = Path(args.label_in)
    image_ref = Path(args.image_ref)
    label_out = Path(args.label_out)
    label_out.mkdir(parents=True, exist_ok=True)

    for lbl_path in sorted(label_in.glob(args.pattern)):
        case = lbl_path.name
        img_path = image_ref / case
        if not img_path.exists():
            raise FileNotFoundError(img_path)

        lbl_img = nib.load(lbl_path)
        lbl = np.asarray(lbl_img.dataobj)

        img = nib.load(img_path)
        target_len = img.shape[args.axis]
        old_len = lbl.shape[args.axis]

        # 最近邻复制 slice（Scheme A 核心）
        idx = np.floor(
            np.linspace(0, old_len - 1, target_len)
        ).astype(np.int64)
        idx = np.clip(idx, 0, old_len - 1)
        lbl_new = np.take(lbl, idx, axis=args.axis)

        # ★ 关键：复制 image 的 header + affine
        header = img.header.copy()
        header.set_data_dtype(lbl_new.dtype)
        header.set_data_shape(lbl_new.shape)

        out_path = label_out / case
        nib.save(
            nib.Nifti1Image(lbl_new.astype(lbl.dtype), img.affine, header),
            out_path
        )

        fg = int((lbl_new > 0).sum())
        print(f"[A] {case}: {lbl.shape} -> {lbl_new.shape}, fg={fg}")

    print("[DONE] Scheme A finished.")


if __name__ == "__main__":
    main()
